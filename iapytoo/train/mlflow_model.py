import mlflow.pyfunc as mp
import numpy as np
from typing import Any
import torch

from torchvision.transforms import ToTensor

from mlflow.models import ModelSignature

from iapytoo.train.valuator import Valuator
from iapytoo.predictions.predictors import Predictor


class MlflowTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, model_input, *args, **kwds) -> torch.Tensor:
        raise NotImplementedError


class TorchTransform(MlflowTransform):

    def __init__(self):
        super().__init__(transform=ToTensor())

    def __call__(self, model_input, *args, **kwds) -> torch.Tensor:
        return self.transform(model_input)


class IMlfowModelProvider:

    def get_input_example(self) -> np.array:
        raise NotImplementedError

    def get_transform(self) -> MlflowTransform:
        return TorchTransform()  # default one, only transforms to torch Tensor

    def get_signature(self) -> ModelSignature:
        # signature is useless if input_example is simple enough
        return None


class MlflowModel(mp.PythonModel):

    def __init__(self) -> None:
        super().__init__()

        self.model = None

        # from IMlfowModelProvider
        self.transform = TorchTransform()  # default one, only transforms to torch Tensor
        self.signature = None
        self.input_example = None

        self.valuator_key = None
        self.predictor_key = None

        # from context
        self.ml_predictor: Predictor = None
        self.ml_valuator: Valuator = None

    def predict(
        self, context: mp.PythonModelContext, model_input: np.ndarray, params: dict[str, Any] | None = None
    ):

        print(f"predict called with input shape: {model_input.shape}")
        model_input_tensor = self.transform(model_input)

        outputs_tensor = self.ml_valuator.evaluate_one(model_input_tensor)

        predictions = self.ml_predictor(outputs_tensor)

        return predictions
