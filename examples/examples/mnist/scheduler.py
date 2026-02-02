
from torch.optim.lr_scheduler import LambdaLR

from iapytoo.utils.config import Config
from iapytoo.train.scheduler import Scheduler


class MnistScheduler(Scheduler):
    def __init__(self, optimizer, config: Config) -> None:
        super().__init__(optimizer, config)

        def lr_lambda(epoch):
            # LR to be 0.1 * (1/1+0.01*epoch)
            return 0.995**epoch

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
