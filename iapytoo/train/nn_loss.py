from torch import Tensor
import torch.nn as nn
from torch.types import Device
from iapytoo.utils.config import Config


class NNLossError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class NNLoss:
    def __init__(self, config: Config) -> None:
        self.torch_loss = None

    def __call__(self, pred, target) -> Tensor:
        return self.torch_loss(pred, target)

    def to(self, device: Device):
        pass


class MSELoss(NNLoss):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.torch_loss = nn.MSELoss()


class NLLLoss(NNLoss):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.torch_loss = nn.NLLLoss()


class CrossEntropyLoss(NNLoss):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.torch_loss = nn.CrossEntropyLoss()
