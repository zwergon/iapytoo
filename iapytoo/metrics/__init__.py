import torch


from iapytoo.utils.config import Config
from iapytoo.utils.singleton import singleton
from .predefined import R2Metric, RMSMetric, MSMetric, AccuracyMetric
from .metric import Metric


class MetricsCollection(Metric):
    def __init__(self, tag: str, metric_names: list[str], config: Config):
        super().__init__(tag, config)
        self.metrics = {}
        factory = MetricFactory()
        try:
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


class MetricError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


@singleton
class MetricFactory:
    def __init__(self) -> None:
        self.metrics_dict = {
            "r2": R2Metric,
            "rms": RMSMetric,
            "ms": MSMetric,
            "accuracy": AccuracyMetric,
        }

    def register_metric(self, key, metric_cls):
        self.metrics_dict[key] = metric_cls

    def create_metric(self, kind: str, config: Config, device="cpu") -> Metric:
        """Creates an architecture of NN

        Args:
            kind (str): kind of metric, key for the factory
            config (dict): config dict to use to initialize metric
            device (str, optional): Defaults to "cpu".

        Raises:
            MetricError: error raised if no architecture fit kind key

        Returns:
            iapytoo.metrics.Metric: metric
        """
        try:
            metric: Metric = self.metrics_dict[kind](config)
        except KeyError:
            raise MetricError(f"metric {kind} is not handled")

        metric = metric.to(device=device)
        return metric
