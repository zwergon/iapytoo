import torch
import logging

import mlflow
import mlflow.pyfunc

from abc import ABC, abstractmethod


from iapytoo.utils.config import Config
from iapytoo.utils.model_config import MLFlowConfig
from iapytoo.train.logger import Logger
from iapytoo.train.valuator import Valuator, ModelValuator
from iapytoo.train.factories import Factory
from iapytoo.train.mlflow_model import MlflowModel, IMlfowModelProvider
from iapytoo.predictions import Predictions, PredictionType
from iapytoo.predictions.predictors import Predictor
from iapytoo.metrics import MetricsCollection


class Inference(ABC):

    def __init__(
        self,
        config: Config
    ) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

        self.valuator = ModelValuator(self.model, self.device)
        self.predictor = Factory().create_predictor(self._config.model.predictor)

        self.mlflow_model = MlflowModel()
        self.mlflow_model.setattr(
            model=self.model,
            valuator=self.valuator,
            predictor_key=self._config.model.predictor,
        )
        if self.mlflow_model_provider is not None:
            self.mlflow_model.signature = self.mlflow_model_provider.get_signature()
            self.mlflow_model.input_example = self.mlflow_model_provider.get_input_example()
            self.mlflow_model.transform = self.mlflow_model_provider.get_transform()

    @abstractmethod
    def predict(self, loader, run_id=None):
        pass


class MLFlowInference(Inference):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def _create_models(self, loader):
        model_config: MLFlowConfig = self._config.model

        logged_model = f"runs:/{model_config.run_id}/model"
        model = mlflow.pyfunc.load_model(
            logged_model).unwrap_python_model().model
        model = model.to(device=self.device)
        return [model]

    def predict(self, loader):

        metrics = MetricsCollection(
            "inference", self._config.metrics, self._config)
        # metrics.to(self.device)

        with Logger(self._config) as self.logger:
            self._display_device()
            self.logger.summary()

            self._models = self._create_models(loader)

            assert self.model is not None, "no model loaded for prediction"

            assert self.predictions is not None, "no predictions defined for this training"
            self.predictions.compute(loader=loader)
            self.logger.report_prediction(0, self.predictions)
            metrics.update(
                self.predictions.tensor(PredictionType.OUTPUTS),
                self.predictions.tensor(PredictionType.ACTUAL),
            )
            metrics.compute()
            self.logger.report_metrics(0, metrics)
