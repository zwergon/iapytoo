import numpy as np
from pathlib import Path

from iapytoo.utils.config import Config
from iapytoo.mlflow.model import MlflowWGANProvider
from iapytoo.predictions.predictors import Predictor
from iapytoo.utils.model_config import GanConfig


from .generator import GruGenerator
from .critic import DFTCritic


class WindProvider(MlflowWGANProvider):

    def __init__(self, config: Config):
        super().__init__(config)
        model_config: GanConfig = config.model
        self._input_example = np.random.rand(model_config.noise_dim)
        self._model = GruGenerator(config)
        self._discriminator = DFTCritic(config)
        self._predictor = Predictor()

    # override
    def code_definition(self) -> dict:
        return {
            "path": str(Path(__file__).parent.parent),
            "provider": {
                "module": "examples.wgan.provider",
                "class": "WindProvider"
            }
        }
