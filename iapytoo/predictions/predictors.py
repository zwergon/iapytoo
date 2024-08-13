
import torch


class Predictor:
    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        # By default, do nothing outputs of the model are predictions.
        return outputs


class MaxPredictor(Predictor):

    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        _, preds = torch.max(outputs.data, 1)
        return preds
