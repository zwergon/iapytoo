from iapytoo.utils.config import Config
from iapytoo.metrics.metric import Metric


class MetricsCollection(Metric):
    def __init__(self, tag: str, metric_names: list[str], config: Config):
        from iapytoo.train.factories import Factory, MetricError
        super().__init__(tag, config)
        self.metrics = {}
        try:
            factory = Factory()
            for n in metric_names:
                self.metrics[f"{tag}_{n}"] = factory.create_metric(n, config)
        except MetricError as er:
            print(f"Unable to create metric : {er}")

    def to(self, device):
        for m in self.metrics.values():
            m.to(device)

    # overwrite
    def update(self, Y_hat, Y):
        for m in self.metrics.values():
            m.update(Y_hat, Y)

    # overwrite
    def compute(self):
        self.results = {}
        for k, m in self.metrics.items():
            results = m.compute()
            for name, result in results.items():
                # prepend the name of the collection to all inner results : validation_accuracy for example
                self.results[f"{self.name}_{name}"] = result
        return self.results

    # overwrite
    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()
