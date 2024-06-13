import os
import torch
import torch.nn as nn

import uuid
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import logging

from threading import Lock

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import TensorSpec, Schema

from iapytoo.utils.config import Config
from iapytoo.utils.display import lrfind_plot
from iapytoo.train.checkpoint import CheckPoint
from iapytoo.predictions import Predictions


class Logger:
    @staticmethod
    def _run_name(root_name):
        return f"{root_name}_{str(uuid.uuid1())[:8]}"

    @property
    def experiment_id(self):
        experiment = mlflow.get_experiment_by_name(self.config["project"])
        if experiment is None:
            return mlflow.create_experiment(self.config["project"])
        else:
            return experiment.experiment_id

    def active_run_name(self):
        active_run = mlflow.active_run()
        if active_run:
            return active_run.info.run_name

        return None

    @property
    def config(self):
        return self._config.__dict__

    def __init__(self, config: Config, run_id: str = None) -> None:
        self._config = config
        self.run_id = run_id
        self.agg = matplotlib.rcParams["backend"]
        self.signature = None
        self.lock = Lock()
        print("tracking_uri", self.config["tracking_uri"])
        if "tracking_uri" in self.config and self.config["tracking_uri"] is not None:
            logging.info(f".. set tracking uri to {self.config['tracking_uri']}")
            mlflow.set_tracking_uri(self.config["tracking_uri"])

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def start(self):
        matplotlib.use("agg")
        params_flag = self.run_id is None
        active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            run_name=self._run_name(self.config["run"]),
        )
        print("active_run", active_run.info.run_id)
        self.run_id = active_run.info.run_id
        if params_flag:
            mlflow.log_params(self._params())

    def close(self):
        matplotlib.use(self.agg)
        mlflow.end_run()

    def set_signature(self, loader):
        X, Y = next(iter(loader))
        x_shape = list(X.shape)
        x_shape[0] = -1
        y_shape = list(Y.shape)
        y_shape[0] = -1

        input_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=x_shape)])
        output_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=y_shape)])
        self.signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    def _params(self):
        # take care, some config parameters are saved by mlflow.
        # When you run it again, these parameters can not change between two runs.
        params = self._config.__dict__.copy()
        if "run_id" in params:
            del params["run_id"]
        del params["epochs"]
        del params["tracking_uri"]
        return params

    def __str__(self):
        msg = "\nLogger:\n"
        network = (
            self.config["model"]
            if "model" in self.config
            else f"{self.config['discriminator']} & {self.config['generator']}"
        )
        msg += f"Network: {network}\n"
        msg += f"matplotlib backend: {matplotlib.rcParams['backend']}, interactive: {matplotlib.is_interactive()}\n"
        msg += f"tracking_uri: {mlflow.get_tracking_uri()}\n"
        active_run = mlflow.active_run()
        if active_run:
            msg += f"Name: {active_run.info.run_name}\n"
            msg += f"Experiment_id: {active_run.info.experiment_id}\n"
            msg += f"Run_id: {self.run_id}\n"
        if self.signature is not None:
            msg += f"Signature {str(self.signature)}\n"

        return msg

    def summary(self):
        logging.info(str(self))
        logging.info(self._config)

    def log_checkpoint(self, checkpoint: CheckPoint):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckp_name = os.path.join(tmpdirname, "checkpoint.pt")
            torch.save(checkpoint.params, ckp_name)
            mlflow.log_artifact(local_path=ckp_name, artifact_path="checkpoints")

    def save_model(self, model: nn.Module):
        with self.lock:
            # model for deployment don't need a GPU device
            # store it on gpu
            device = torch.device("cpu")
            mlflow.pytorch.log_model(
                model.to(device),
                "model",
                signature=self.signature,
                extra_pip_requirements=["--extra-index-url https://zwergon.github.io"],
            )

    def report_metric(self, epoch, metrics: dict):
        with self.lock:
            mlflow.log_metrics(metrics, step=epoch)

    def report_metrics(self, epoch, metrics):
        with self.lock:
            values = {}
            for k, v in metrics.results.items():
                if len(v.shape) == 0:
                    values[k] = v.item()
                else:
                    for c in range(v.shape[0]):
                        values[f"{k}_{c}"] = v[c].item()
            mlflow.log_metrics(values, step=epoch)

    def report_prediction(self, epoch, predictions: Predictions):
        with self.lock:
            plots = predictions.plot(epoch)
            for name, fig in plots.items():
                if fig is not None:
                    mlflow.log_figure(fig, artifact_file=f"{name}/{name}_{epoch}.png")
                    plt.close(fig)

    def report_findlr(self, lrs, losses):
        fig = lrfind_plot(lrs, losses)
        with self.lock:
            mlflow.log_figure(figure=fig, artifact_file="find_lr.jpg")
        plt.close(fig)
