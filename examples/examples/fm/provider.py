import numpy as np

from iapytoo.mlflow.model import MlflowModelProvider
from iapytoo.train.model import FlowMatchingModel
from iapytoo.utils.config import Config


class SineFMProvider(MlflowModelProvider):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # The backbone named by config.model.backbone must be registered first.
        self._model = FlowMatchingModel(config)
        # One example of shape (C, L), with no batch axis.
        self._input_example = np.random.randn(
            config.model.n_channels, config.model.signal_length).astype(np.float32)

    def code_definition(self) -> dict:
        from pathlib import Path
        return {
            "path": str(Path(__file__).parent),
            "provider": {
                "module": "examples.fm.provider",
                "class": "SineFMProvider"
            }
        }
