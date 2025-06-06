import mlflow.pyfunc as mp
import numpy as np
from typing import Any
import torch

from torchvision.transforms import ToTensor

from mlflow.models import ModelSignature


from iapytoo.train.factories import Factory
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
        self.predictor_key: str = None
        self.valuator_key: str = None

        # from IMlfowModelProvider
        self.transform = TorchTransform()  # default one, only transforms to torch Tensor
        self.signature = None
        self.input_example = None

        # from context
        self.predictor: Predictor = None
        self.valuator: Valuator = None

    def load_context(self, context: mp.PythonModelContext):
        # Prédire avec le modèle sur cpu
        factory = Factory()
        self.valuator = factory.create_valuator(
            self.valuator_key,
            self.model,
            "cpu"
        )
        self.predictor = factory.create_predictor(
            self.predictor_key
        )

    def predict(
        self, context: mp.PythonModelContext, model_input: np.ndarray, params: dict[str, Any] | None = None
    ):
        print(f"predict called with input shape: {model_input.shape}")

        model_input_tensor = self.transform(model_input)

        outputs_tensor = self.valuator.evaluate_one(model_input_tensor)

        predicted_tensor = self.predictor(outputs_tensor)

        return predicted_tensor.numpy()
