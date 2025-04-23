import torch

from iapytoo.utils.config import Config
from iapytoo.predictions.predictors import PredictorFactory


class Metric:
    def __init__(self, name, config: Config, with_target=True):
        self.with_target = with_target
        self.name = name
        self.config = config
        self.predictor = PredictorFactory().create_predictor(config)

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
        return self.predictor(self.outputs) if self.predictor else self.outputs

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

    def compute(self):
        pass
