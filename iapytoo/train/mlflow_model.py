
import numpy as np
import logging
from typing import Any

import torch

import mlflow.pyfunc as mp

from iapytoo.train.factories import Factory
from iapytoo.train.valuator import Valuator
from iapytoo.predictions.predictors import Predictor


class MlflowTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, model_input, *args, **kwds) -> np.array:
        raise NotImplementedError


class IMlfowModelProvider:

    def get_input_example(self) -> np.array:
        return None  # by default no input example

    def get_transform(self) -> MlflowTransform:
        return None  # by default no transform


class _MlflowModelPrivate:

    @staticmethod
    def from_context(mlflow_model: '_MlflowModelPrivate', context: mp.PythonModelContext):
        private = _MlflowModelPrivate()

        saved_model = torch.load(
            context.artifacts["model"], weights_only=False)

        if 'model' in saved_model:
            private.model = saved_model['model']

        if "transform" in saved_model:
            private.transform = saved_model['transform']

        factory = Factory()
        private.valuator = factory.create_valuator(
            mlflow_model.metadata["valuator_key"],
            private.model,
            "cpu"
        )
        private.predictor = factory.create_predictor(
            mlflow_model.metadata['predictor_key']
        )

        if mlflow_model.metadata.get('inference_key') is not None:
            private.ml_predictor = factory.create_predictor(
                mlflow_model.metadata['inference_key'],
                mlflow_model.metadata.get('inference_args', {})
            )
        else:
            private.ml_predictor = private.predictor

        mlflow_model._private = private

    def __init__(self):
        self.model = None
        self.transform = None
        self.valuator: Valuator = None
        self.predictor: Predictor = None
        self.ml_predictor: Predictor = None


class MlflowModel(mp.PythonModel):

    INPUT_EXAMPLE = "input_example"

    def __init__(self, metadata: dict = None) -> None:
        super().__init__()
        self.metadata = metadata
        self._private: _MlflowModelPrivate = None

    @property
    def transform(self):
        return self._private.transform

    @property
    def model(self):
        return self._private.model

    @property
    def valuator(self):
        return self._private.valuator

    @property
    def predictor(self):
        return self._private.predictor

    @property
    def ml_predictor(self):
        return self._private.ml_predictor

    def load_context(self, context):
        super().load_context(context)
        _MlflowModelPrivate.from_context(self, context)

    def predict(
        self,
        context: mp.PythonModelContext,
        model_input: list[str | np.ndarray],
        params: dict[str, Any] | None = None
    ):
        if not isinstance(model_input, (list, tuple)):
            raise ValueError("model_input must be a list of file paths")

        arrays = []
        for path in model_input:

            if isinstance(path, np.ndarray):
                arr = path.astype(np.float32)
            elif isinstance(path, str):
                if path == MlflowModel.INPUT_EXAMPLE:
                    assert MlflowModel.INPUT_EXAMPLE in context.artifacts, "no input example given during training"
                    arr = np.load(context.artifacts[MlflowModel.INPUT_EXAMPLE])
                else:
                    arr = np.load(path)
            else:
                raise TypeError("Unsupported input type")

            arrays.append(arr)

            batch = np.stack(arrays, axis=0).astype(np.float32)

        logging.info(f"predict called with input shape: {batch.shape}")
        if self.transform is not None:
            logging.info("predict use a transform")
            batch = self.transform(batch)

        batch_tensor = torch.from_numpy(batch)

        outputs_tensor = self.valuator.evaluate_one(batch_tensor)

        predictions = self.ml_predictor(outputs_tensor)

        return predictions
