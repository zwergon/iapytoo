import torch

from iapytoo.utils.config import Config
from iapytoo.predictions.predictors import Predictor


class MetricError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Metric:
    def __init__(self, name, config: Config, predictor: Predictor = None, with_target=True):
        self.with_target = with_target
        self.name = name
        self.config = config
        self.predictor = predictor

        # First dimension is for the concatenation.
        self.outputs = torch.zeros(size=(0,))
        if self.with_target:
            self.target = torch.zeros(size=(0,))
        self.results = {}

    @property
    def device(self):
        return self.outputs.device

    @property
    def predicted(self):
        assert self.predictor is not None, "need a predictor to compute metric"
        return self.predictor(self.outputs)

    def to(self, device):
        self.outputs = self.outputs.to(device)
        if self.with_target:
            self.target = self.target.to(device)
        return self

    def update(self, outputs, Y):
        self.outputs = torch.cat((self.outputs, outputs), dim=0)
        if self.with_target:
            self.target = torch.cat((self.target, Y), dim=0)

    def reset(self):
        device = self.device
        self.outputs = torch.zeros(size=(0,), device=device)
        if self.with_target:
            self.target = torch.zeros(size=(0,), device=self.device)

    def gather(self):
        import torch.distributed as dist
        if dist.is_initialized():
            world_size = dist.get_world_size()

            gathered_outputs = [torch.zeros_like(
                self.outputs) for _ in range(world_size)]
            dist.all_gather(gathered_outputs, self.outputs)

            if self.with_target:
                gathered_targets = [torch.zeros_like(
                    self.target) for _ in range(world_size)]
                dist.all_gather(gathered_targets, self.target)

            self.outputs = torch.cat(gathered_outputs, dim=0)
            if self.with_target:
                self.target = torch.cat(gathered_targets, dim=0)

    def compute(self):
        pass
