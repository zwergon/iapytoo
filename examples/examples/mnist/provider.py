from pathlib import Path

import torch
import numpy as np
from iapytoo.dataset.scaling import MeanNormalize
from iapytoo.dataset.transform import Transform
from iapytoo.predictions.predictors import MaxPredictor, Predictor
from iapytoo.train.mlflow_model import MlflowModelProvider
from iapytoo.utils.config import Config
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from .model import MnistModel


class MnistProvider(MlflowModelProvider):
    def __init__(self, config: Config):
        super().__init__(config)

        self._model = MnistModel(config)
        self._transform: Transform = lambda x: torch.from_numpy(MeanNormalize(config)(x))
        self._predictor: Predictor = MaxPredictor()

    def get_input_example(self):

        return np.random.rand(1, 1, 28, 28).astype(np.float32)

    @property
    def _signature(self):
        # --- np input Array signature
        X = self.get_input_example()
        input_schema = Schema([TensorSpec(type=X.dtype, shape=X.shape)])
        output_schema = Schema([TensorSpec(type=np.dtype("int64"), shape=(1,))])

        return ModelSignature(inputs=input_schema, outputs=output_schema)

    # override
    def code_definition(self) -> dict:

        return {
            "path": str(Path(__file__).parent.parent),
            "provider": {"module": "examples.mnist.provider", "class": "MnistProvider"},
        }
