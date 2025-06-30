import torch
import logging

import numpy as np

import mlflow
import mlflow.pyfunc

from mlflow.models import ModelSignature

from abc import ABC, abstractmethod


from iapytoo.utils.config import Config
from iapytoo.utils.model_config import ModelConfig, MLFlowConfig
from iapytoo.train.logger import Logger
from iapytoo.train.factories import Factory
from iapytoo.train.mlflow_model import MlflowTransform, MlflowModel, IMlfowModelProvider
from iapytoo.predictions import Predictions, PredictionType
from iapytoo.metrics.collection import MetricsCollection


class Inference(ABC, IMlfowModelProvider):

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
        self.mlflow_model_provider: IMlfowModelProvider = None

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

    @abstractmethod
    def _create_models(self, loader):
        pass

    def _init_mlflow_model(self):
        """ Generates mlflow model by creating valuator and predictor for this inference
            if a IMlfowModelProvider is provided, use it to initiate signature, transform and input_example
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

        self.mlflow_model = MlflowModel()
        self.mlflow_model.model = self.model
        self.mlflow_model.valuator_key = model_config.valuator
        self.mlflow_model.predictor_key = model_config.predictor

        if self.mlflow_model_provider is not None:
            self.mlflow_model.signature = self.mlflow_model_provider.get_signature()
            self.mlflow_model.input_example = self.mlflow_model_provider.get_input_example()
            self.mlflow_model.transform = self.mlflow_model_provider.get_transform()

    def get_transform(self) -> MlflowTransform:
        if self.mlflow_model_provider is not None:
            return self.mlflow_model_provider.get_transform()
        return None

    def get_input_example(self) -> np.array:
        if self.mlflow_model_provider is not None:
            return self.mlflow_model_provider.get_input_example()
        return None

    def get_signature(self) -> ModelSignature:
        if self.mlflow_model_provider is not None:
            return self.mlflow_model_provider.get_signature()
        return None

    @abstractmethod
    def predict(self, loader, run_id=None):
        pass


class MLFlowInference(Inference):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        model_config: MLFlowConfig = self._config.model
        logged_model = f"runs:/{model_config.run_id}/model"
        self.mlflow_model: MlflowModel = mlflow.pyfunc.load_model(
            logged_model).unwrap_python_model()

    # overidde : use mlflow_model
    def get_transform(self) -> MlflowTransform:
        if self.mlflow_model is not None:
            return self.mlflow_model.transform
        return None

    # overidde : use mlflow_model
    def get_input_example(self) -> np.array:
        if self.mlflow_model is not None:
            return self.mlflow_model.input_example
        return None

    # overidde : use mlflow_model
    def get_signature(self) -> ModelSignature:
        if self.mlflow_model is not None:
            return self.mlflow_model.signature
        return None

    # overidde : use mlflow_model
    def _create_models(self, loader):
        model = self.mlflow_model.model
        model.to(self.device)
        return [model]

    def predict(self, loader):

        metrics = MetricsCollection(
            "inference", self._config.metrics.names, self._config)
        # metrics.to(self.device)

        with Logger(self._config) as self.logger:
            self._display_device()
            self.logger.summary()

            self._models = self._create_models(loader)
            assert self.model is not None, "no model loaded for prediction"

            factory = Factory()
            self.valuator = factory.create_valuator(
                self.mlflow_model.valuator_key,
                self.model,
                self.device
            )
            self.predictor = factory.create_predictor(
                self.mlflow_model.predictor_key
            )

            self.predictions.compute(loader=loader)
            self.logger.report_prediction(0, self.predictions)
            metrics.update(
                self.predictions.tensor(PredictionType.OUTPUTS),
                self.predictions.tensor(PredictionType.ACTUAL),
            )
            metrics.compute()
            self.logger.report_metrics(0, metrics)
