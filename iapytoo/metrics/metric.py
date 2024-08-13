import torch


class Metric:
    def __init__(self, name, config, with_target=True):
        self.with_target = with_target
        self.name = name
        self.config = config

        # First dimension is for the concatenation.
        self.predicted = torch.zeros(size=(0,))
        if self.with_target:
            self.target = torch.zeros(size=(0,))
        self.results = {}

    @property
    def device(self):
        return self.predicted.device

    def to(self, device):
        self.predicted = self.predicted.to(device)
        if self.with_target:
            self.target = self.target.to(device)
        return self

    def update(self, Y_hat, Y):
        self.predicted = torch.cat((self.predicted, Y_hat), dim=0)
        if self.with_target:
            self.target = torch.cat((self.target, Y), dim=0)

    def reset(self):
        device = self.device
        self.predicted = torch.zeros(size=self.output_size, device=device)
        if self.with_target:
            self.target = torch.zeros(
                size=self.output_size, device=self.device)

    def compute(self):
        pass
