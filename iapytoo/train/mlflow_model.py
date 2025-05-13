import mlflow.pyfunc as mp
import numpy as np
from typing import Any
from torch.utils.data import DataLoader

from iapytoo.train.valuator import Valuator, ModelValuator
from iapytoo.predictions.predictors import Predictor, PredictorFactory


class Transform:

    @staticmethod
    def img_to_numpy(img):
        X = np.array(img)
        if len(X.shape) == 3:  # Vérifier si l'image est en couleur
            return X.transpose(2, 0, 1)
        else:
            return np.expand_dims(img, axis=0).astype(np.float32)

    def __init__(self, train_transform, dataset=None, infer_transform=None) -> None:
        self.train_transform = train_transform
        self.infer_transform = infer_transform if infer_transform is not None else train_transform
        self.set_signature(dataset)

    def transform(self, context, model_input, params=None):
        return model_input

    def get_signature(self):
        if self.signature is None or self.input_example is None:
            raise Exception(
                "The signature should be set. It is model-dependent")
        return self.signature, self.input_example

    def set_signature(self, dataset):
        self.signature = None
        self.input_example = None
        raise NotImplementedError


class ModelWrapper(mp.PythonModel):

    def __init__(self) -> None:
        super().__init__()
        self.transform: Transform = None
        self.model = None
        self.signature = None
        self.input_example = None
        self.loader: DataLoader = None
        self.predictor_key: str = None
        self.predictor: Predictor = None
        self.valuator: Valuator = None

    def load_context(self, context: mp.PythonModelContext):
        # Prédire avec le modèle sur cpu
        self.valuator = ModelValuator(self.model, "cpu")
        self.predictor = PredictorFactory().create_predictor(self.predictor_key)

    def predict(
        self, context: mp.PythonModelContext, model_input: np.ndarray, params: dict[str, Any] | None = None
    ):
        print(f"predict called with input shape: {model_input.shape}")

        model_input_tensor = self.transform.transform(
            context, model_input, params)

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

        self.signature, self.input_example = self.transform.get_signature()
