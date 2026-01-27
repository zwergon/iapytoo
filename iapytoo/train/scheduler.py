
from torch.optim.lr_scheduler import StepLR

from iapytoo.utils.config import Config


class SchedulerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Scheduler:
    def __init__(self, optimizer, config: Config) -> None:
        self.lr_scheduler = None

    def update(self, loss: float):
        pass

    def step(self):
        assert self.lr_scheduler is not None, f"no torch scheduler associated with {self.__class__.__name__}"
        self.lr_scheduler.step()


class StepScheduler(Scheduler):
    def __init__(self, optimizer, config) -> None:
        super().__init__(optimizer, config)

        kwargs = {}
        kwargs["gamma"] = config.training.gamma
        kwargs["step_size"] = config.training.step_size
        self.lr_scheduler = StepLR(optimizer, **kwargs)
