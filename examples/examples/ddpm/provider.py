import torch
import numpy as np


from iapytoo.utils.config import Config
from iapytoo.train.mlflow_model import MlflowModelProvider
from iapytoo.utils.model_config import DDPMConfig

from .models.unet import UNet1D


def check_shapes(model, B=2, C=1, L=32):
    x = torch.randn(B, C, L)
    t = torch.randn(B)
    out = model(x, t)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    return out


class PSDProvider(MlflowModelProvider):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._model = UNet1D(config)
        ddpm_config: DDPMConfig = config.model
        self._input_example = np.random.rand(1, ddpm_config.signal_length)
        check_shapes(self._model)

    def code_definition(self) -> dict:
        from pathlib import Path
        return {
            "path": str(Path(__file__).parent),
            "provider": {
                "module": "wind_diff.provider",
                "class": "PSDProvider"
            }
        }
