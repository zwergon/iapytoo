from typing import Any
from torchmetrics.metric import Metric


class MetricsCollection(Metric):
    def __init__(self, tag: str, metric_creators: list, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metrics = {f"{tag}_{c.name}": c.create() for c in metric_creators}
        self.results = {}

    def to(self, device):
        for m in self.metrics.values():
            m.to(device)

    # overwrite
    def update(self, Y_hat, Y):
        for b in range(Y.shape[0]):
            for m in self.metrics.values():
                m.update(Y_hat, Y)

    # overwrite
    def compute(self):
        self.results = {k: m.compute() for k, m in self.metrics.items()}
        return self.results

    # overwrite
    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()
