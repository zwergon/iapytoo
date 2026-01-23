from iapytoo.utils.config import Config


class NNLossError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class NNLoss:
    def __init__(self, config: Config) -> None:
        self.torch_loss = None

    def __call__(self, pred, target):
        return self.torch_loss(pred, target)
