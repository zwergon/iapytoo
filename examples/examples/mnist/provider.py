import numpy as np
from pathlib import Path

from iapytoo.utils.config import Config
from iapytoo.train.mlflow_model import MlflowModelProvider
from iapytoo.dataset.transform import Transform
from iapytoo.dataset.scaling import MeanNormalize
from iapytoo.predictions.predictors import Predictor, MaxPredictor

from .model import MnistModel


class MnistProvider(MlflowModelProvider):

    def __init__(self, config: Config):
        super().__init__(config)

        self._model = MnistModel(config)
        self._input_example = np.random.rand(1, 28, 28)
        self._transform: Transform = MeanNormalize(config)
        self._predictor: Predictor = MaxPredictor()

    # override
    def code_definition(self) -> dict:

        return {
            "path": str(Path(__file__).parent.parent),
            "provider": {
                "module": "examples.mnist.provider",
                "class": "MnistProvider"
            }
        }
