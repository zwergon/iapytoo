import mlflow.pyfunc as mp
import numpy as np
from typing import Any

from iapytoo.train.valuator import Valuator, ModelValuator
from iapytoo.predictions.predictors import Predictor, PredictorFactory


class Transform:

    def __init__(self, train_transform, infer_transform=None) -> None:
        self.train_transform = train_transform
        self.infer_transform = infer_transform if infer_transform is not None else train_transform

    def transform(self, context, model_input, params=None):
        return model_input


class ModelWrapper(mp.PythonModel):

    def __init__(self) -> None:
        super().__init__()
        self.transform: Transform = None
        self.model = None
        self.signature = None
        self.input_example = None
        self.predictor_key: str = None
        self.predictor: Predictor = None
        self.valuator: Valuator = None

    def load_context(self, context: mp.PythonModelContext):
        # Prédire avec le modèle sur cpu
        self.valuator = ModelValuator(self.model, 'cpu')
        self.predictor = PredictorFactory().create_predictor(self.predictor_key)

    def predict(self,
                context: mp.PythonModelContext,
                model_input: np.ndarray,
                params: dict[str, Any] | None = None):
        print(f"predict called with input shape: {model_input.shape}")

        model_input_tensor = self.transform.transform(
            context, model_input, params)

        outputs_tensor = self.valuator.evaluate_one(model_input_tensor)

        predicted_tensor = self.predictor(outputs_tensor)

        return predicted_tensor.numpy()
