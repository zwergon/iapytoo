import mlflow

from iapytoo.utils.config import Config
from iapytoo.train.logger import Logger
from iapytoo.utils.model_config import MLFlowConfig
from iapytoo.predictions import Predictor, PredictionType
from iapytoo.train.training import Inference
from iapytoo.metrics import MetricsCollection


class MLFlowInference(Inference):

    def __init__(
        self,
        config: Config,
        predictor: Predictor = Predictor()
    ) -> None:
        super().__init__(config, predictor=predictor)

    def _create_models(self, loader):
        model_config: MLFlowConfig = self._config.model

        logged_model = f"runs:/{model_config.run_id}/model"
        model = mlflow.pytorch.load_model(logged_model)
        model = model.to(device=self.device)
        return [model]

    def predict(self, loader):

        metrics = MetricsCollection("inference", self._config)
        # metrics.to(self.device)

        with Logger(self._config) as self.logger:
            self._display_device()
            self.logger.summary()

            self._models = self._create_models(loader)

            assert self.model is not None, "no model loaded for prediction"

            assert self.predictions is not None, "no predictions defined for this training"
            self.predictions.compute(valuator=self._valuator(loader))
            self.logger.report_prediction(0, self.predictions)
            metrics.update(self.predictions.tensor(PredictionType.OUTPUTS),
                           self.predictions.tensor(PredictionType.ACTUAL))
            metrics.compute()
            self.logger.report_metrics(0, metrics)
