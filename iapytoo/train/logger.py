import os
import torch

import uuid
import tempfile
import matplotlib.pyplot as plt
import matplotlib
import logging

from threading import Lock

import mlflow
import mlflow.pyfunc


from iapytoo.utils.config import Config
from iapytoo.utils.display import lrfind_plot
from iapytoo.train.checkpoint import CheckPoint
from iapytoo.train.context import Context
from iapytoo.predictions import Predictions
from iapytoo.train.mlflow_model import MlflowModel


class Logger:
    @staticmethod
    def _run_name(root_name):
        return f"{root_name}_{str(uuid.uuid1())[:8]}"

    @property
    def experiment_id(self):
        experiment = mlflow.get_experiment_by_name(self.config.project)
        if experiment is None:
            return mlflow.create_experiment(self.config.project)
        else:
            return experiment.experiment_id

    @property
    def artifact_uri(self):
        if self.run_id is None:
            return None
        run = mlflow.get_run(self.run_id)
        assert run is not None, f"unable to find run {self.run_id}"
        return run.info.artifact_uri

    def active_run_name(self):
        active_run = mlflow.active_run()
        if active_run:
            return active_run.info.run_name

        return None

    def __init__(self, config: Config, run_id: str = None) -> None:
        self.config = config
        self.run_id = run_id
        self.agg = matplotlib.rcParams["backend"]
        self.lock = Lock()
        if self.config.tracking_uri is not None:
            logging.info(f".. set tracking uri to {self.config.tracking_uri}")
            mlflow.set_tracking_uri(self.config.tracking_uri)
        else:
            logging.info("no tracking_uri set")

        self.context = Context(run_id)

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
            run_name=self._run_name(self.config.run),
        )
        logging.info(f"active_run {active_run.info.run_id}")
        self.run_id = active_run.info.run_id
        if params_flag:
            mlflow.log_params(self._params())

    def close(self):
        matplotlib.use(self.agg)
        mlflow.end_run()

    def _params(self):
        # take care, some config parameters are saved by mlflow.
        # When you run it again, these parameters can not change between two runs.
        params = self.config.to_flat_dict(exclude_unset=True)

        for key in [
            "training.epochs",
            "project",
            "run",
            "tracking_uri",
        ]:
            if key in params:
                del params[key]

        return params

    def __str__(self):
        msg = "\nLogger:\n"

        model_config = self.config.model

        msg += f"Network: {model_config._network()}\n"
        msg += f"matplotlib backend: {matplotlib.rcParams['backend']}, interactive: {matplotlib.is_interactive()}\n"
        msg += f"tracking_uri: {mlflow.get_tracking_uri()}\n"
        active_run = mlflow.active_run()
        if active_run:
            msg += f"Name: {active_run.info.run_name}\n"
            msg += f"Experiment_id: {active_run.info.experiment_id}\n"
            msg += f"Run_id: {self.run_id}\n"

        return msg

    def summary(self):
        logging.info(str(self))
        logging.info(self.config)

    def set_epoch(self, epoch):
        self.context.epoch = epoch
        self._save_context()

    def _save_context(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ctx_path = self.context.save(tmpdirname)
            mlflow.log_artifact(ctx_path)

    def log_checkpoint(self, checkpoint: CheckPoint):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckp_name = os.path.join(tmpdirname, "checkpoint.pt")
            torch.save(checkpoint.params, ckp_name)
            mlflow.log_artifact(local_path=ckp_name,
                                artifact_path="checkpoints")

    def save_model(self, model_wrapper: MlflowModel):
        assert model_wrapper is not None, "no model_wrapper instance ?"

        signature = model_wrapper.signature
        input_example = model_wrapper.input_example

        model_wrapper.model.cpu()

        with self.lock:
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=model_wrapper,
                signature=signature,
                input_example=input_example,
                extra_pip_requirements=self.config.training.inference_pip_requirements,
                code_paths=self.config.training.inference_extra_paths,
                conda_env=None,
            )

    def can_report(self):
        return self.context.epoch > self.context.last_epoch

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
                    mlflow.log_figure(
                        fig, artifact_file=f"{name}/{name}_{epoch}.png")
                    plt.close(fig)

    def report_findlr(self, lrs, losses):
        fig = lrfind_plot(lrs, losses)
        with self.lock:
            mlflow.log_figure(figure=fig, artifact_file="find_lr.jpg")
        plt.close(fig)
