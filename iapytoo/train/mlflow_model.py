import mlflow.pyfunc as mp
import numpy as np
from typing import Any
from torch.utils.data import DataLoader

from iapytoo.train.factories import Factory
from iapytoo.train.valuator import Valuator, ModelValuator
from iapytoo.predictions.predictors import Predictor


class MlflowTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, model_input, *args, **kwds):
        return model_input


class IMlfowModelProvider:

    def get_transform(self) -> MlflowTransform:
        raise NotImplementedError

    def get_input_example(self):
        raise NotImplementedError

    def get_signature(self):
        raise NotImplementedError


class MlflowModel(mp.PythonModel):

    def __init__(self) -> None:
        super().__init__()
        self.transform = None
        self.model = None
        self.signature = None
        self.input_example = None
        self.predictor_key: str = None
        self.predictor: Predictor = None
        self.valuator: Valuator = None

    def load_context(self, context: mp.PythonModelContext):
        # Prédire avec le modèle sur cpu
        self.valuator = ModelValuator(self.model, "cpu")
        self.predictor = Factory().create_predictor(self.predictor_key)

    def predict(
        self, context: mp.PythonModelContext, model_input: np.ndarray, params: dict[str, Any] | None = None
    ):
        print(f"predict called with input shape: {model_input.shape}")

        model_input_tensor = self.transform(model_input)

        outputs_tensor = self.valuator.evaluate_one(model_input_tensor)

        predicted_tensor = self.predictor(outputs_tensor)

        return predicted_tensor.numpy()

    def setattr(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise Exception(
                    f'The wrapper class does not have the attribute "{param}"')

        if self.transform is not None:
            self.signature, self.input_example = self.transform.get_signature()
