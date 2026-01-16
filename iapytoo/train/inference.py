import torch
import logging

import numpy as np

import mlflow
import mlflow.pyfunc

from mlflow.tracking import MlflowClient

from abc import ABC, abstractmethod


from iapytoo.utils.config import Config
from iapytoo.utils.model_config import ModelConfig, MLFlowConfig
from iapytoo.train.logger import Logger
from iapytoo.train.factories import Factory
from iapytoo.train.mlflow_model import MlflowModel, MlflowModelProvider
from iapytoo.predictions import Predictions, PredictionType
from iapytoo.metrics.collection import MetricsCollection


class Inference(ABC):

    def __init__(
        self,
        config: Config
    ) -> None:

        self.device = "cuda" if torch.cuda.is_available() and config.cuda else "cpu"
        self._config: Config = config
        self.logger: Logger = None

        self._models = []

        self.predictions: Predictions = Predictions(self)
        self.mlflow_model_provider: MlflowModelProvider = self._create_model_provider()

    def _display_device(self):
        use_cuda = torch.cuda.is_available()
        if self._config.cuda and use_cuda:
            msg = "\n__CUDA\n"
            msg += f"__CUDNN VERSION: {torch.backends.cudnn.version()}\n"
            msg += f"__Number CUDA Devices: {torch.cuda.device_count()}\n"
            msg += f"__CUDA Device Name: {torch.cuda.get_device_name(0)}\n"
            msg += f"__CUDA Device Total Memory [GB]: {torch.cuda.get_device_properties(0).total_memory / 1e9}\n"
            msg += "-----------\n"
            logging.info(msg)
        else:
            logging.info("__CPU")

    @property
    def model(self):
        return self._models[0]

    @property
    def predictor(self):
        assert self.mlflow_model_provider is not None, "a provider is required to define predictor"
        return self.mlflow_model_provider.predictor

    @property
    def transform(self):
        if self.mlflow_model_provider is None:
            return None
        return self.mlflow_model_provider.transform

    def _create_model_provider(self) -> MlflowModelProvider:
        model_config: ModelConfig = self._config.model
        assert model_config.provider is not None, "a provider is need to define model"
        return Factory().create_provider(
            model_config.provider, self._config
        )

    @abstractmethod
    def _create_models(self):
        pass

    @abstractmethod
    def predict(self, loader, run_id=None):
        pass


def get_model_uri(run_id, idx=-1):
    """return the idx model_uri for this run_id.
    return the last one by default.
    """
    client = MlflowClient()
    run = client.get_run(run_id=run_id)
    model_uri = f"models:/{run.outputs.model_outputs[-1].model_id}"
    return model_uri


class MLFlowInference(Inference):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        model_config: MLFlowConfig = self._config.model

        model_uri = get_model_uri(model_config.run_id)
        self.mlflow_model: MlflowModel = mlflow.pyfunc.load_model(
            model_uri).unwrap_python_model()

    def _create_models(self):
        model = self.mlflow_model.model
        model.to(self.device)
        return [model]

    def predict(self, loader):

        metrics = MetricsCollection(
            "inference",
            self._config.metrics.names,
            self._config,
            predictor=self.predictor)

        with Logger(self._config) as self.logger:
            self._display_device()
            self.logger.summary()

            self._models = self._create_models()

            self.predictions.compute(loader=loader)
            self.logger.report_prediction(0, self.predictions)
            metrics.update(
                self.predictions.tensor(PredictionType.OUTPUTS),
                self.predictions.tensor(PredictionType.ACTUAL),
            )
            metrics.compute()
            self.logger.report_metrics(0, metrics)
