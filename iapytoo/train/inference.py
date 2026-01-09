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
from iapytoo.train.mlflow_model import MlflowTransform, MlflowModel, MlfowModelProvider
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
        self.valuator = None
        self.predictor = None
        self.predictions: Predictions = Predictions(self)
        self.mlflow_model: MlflowModel = None
        self.mlflow_model_provider: MlfowModelProvider = self.create_mlflow_provider(
            config)

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
    def transform(self):
        if self.mlflow_model_provider is None:
            return None
        return self.mlflow_model_provider.transform

    @abstractmethod
    def _create_models(self, loader):
        pass

    def create_mlflow_provider(self, config: Config) -> MlfowModelProvider:
        return None

    def _init_mlflow_model(self):
        """ Generates mlflow model by creating valuator and predictor for this inference
            if a MlfowModelProvider is provided, use it to initiate signature, transform and input_example
            Take care that at that point model should already be created
        """

        assert self.model is not None, "a pytorch model is required for mlflow model"

        model_config: ModelConfig = self._config.model

        factory = Factory()
        self.valuator = factory.create_valuator(
            model_config.valuator,
            self.model,
            self.device
        )
        self.predictor = factory.create_predictor(
            model_config.predictor
        )

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

    def _create_models(self, loader):
        model = self.mlflow_model.model
        model.to(self.device)
        return [model]

    def predict(self, loader):

        metrics = MetricsCollection(
            "inference",
            self._config.metrics.names,
            self._config)
        # metrics.to(self.device)

        with Logger(self._config) as self.logger:
            self._display_device()
            self.logger.summary()

            self._models = self._create_models(loader)
            assert self.model is not None, "no model loaded for prediction"

            self.valuator = self.mlflow_model.valuator
            self.valuator.device = self.device
            self.predictor = self.mlflow_model.predictor

            self.predictions.compute(loader=loader)
            self.logger.report_prediction(0, self.predictions)
            metrics.update(
                self.predictions.tensor(PredictionType.OUTPUTS),
                self.predictions.tensor(PredictionType.ACTUAL),
            )
            metrics.compute()
            self.logger.report_metrics(0, metrics)
