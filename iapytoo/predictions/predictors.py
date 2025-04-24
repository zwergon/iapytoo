import torch

from iapytoo.utils.singleton import singleton
from iapytoo.utils.config import Config


class Predictor:
    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        # By default, do nothing outputs of the model are predictions.
        return outputs


class MaxPredictor(Predictor):
    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        _, preds = torch.max(outputs.data, 1)
        return preds


@singleton
class PredictorFactory:
    def __init__(self) -> None:
        self.predictor_dict = {"max": MaxPredictor}

    def create_predictor(self, config: Config):
        kind = config.model.predictor
        if kind is not None:
            return self.predictor_dict[kind]()

        return None
