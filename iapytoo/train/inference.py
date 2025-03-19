import torch


import mlflow

from iapytoo.utils.config import Config
from iapytoo.train.logger import Logger
from iapytoo.train.checkpoint import CheckPoint
from iapytoo.utils.model_config import MLFlowConfig
from iapytoo.predictions import Predictions, Predictor, Valuator
from iapytoo.train.training import Inference


class MLFlowInference(Inference):

    def __init__(
        self,
        config: Config,
        predictor: Predictor = Predictor()
    ) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._config = config
        self.logger = None
        self.predictions = Predictions(predictor)

        self._model = None

    def _valuator(self, loader):
        return MLFlowValuator(self, loader)

    def _create_models(self, loader):
        model_config: MLFlowConfig = self._config.model

        logged_model = f"runs:/{model_config.run_id}/model"
        model = mlflow.pytorch.load_model(logged_model)
        return [model]

    def predict(self, loader):

        model_config: MLFlowConfig = self._config.model

        with Logger(self._config, run_id=model_config.run_id) as self.logger:
            self._display_device()
            self.logger.summary()

            self._models = self._create_models(loader)

            assert self.model is not None, "no model loaded for prediction"

            assert self.predictions is not None, "no predictions defined for this training"
            self.predictions.compute(valuator=self._valuator(loader))
            self.logger.report_prediction(0, self.predictions)


class MLFlowValuator(Valuator):

    def __init__(self, inference: MLFlowInference, loader):
        super().__init__(loader, device=inference.device)
        self.inference: MLFlowInference = inference

    def evaluate(self):
        model = self.inference.model
        for X, Y in self.loader:
            X = X.to(self.device)
            model_output = model.predict(X)

            outputs = model_output.detach().cpu()
            actual = Y.detach().cpu()
            yield outputs, actual
