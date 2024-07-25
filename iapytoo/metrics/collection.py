import torch
from iapytoo.utils.meta_singleton import MetaSingleton


class Metric:
    def __init__(self, name, config, loader, with_target=True):
        self.with_target = with_target
        self.name = name
        X, Y = next(iter(loader))
        self.output_size = (0,) + tuple(Y.shape)

        self.predicted = torch.zeros(size=self.output_size)
        if self.with_target:
            self.target = torch.zeros(size=self.output_size)
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
        self.predicted = torch.cat((self.predicted, Y_hat.view(1, -1)), dim=0)
        if self.with_target:
            self.target = torch.cat((self.target, Y.view(1, -1)), dim=0)

    def reset(self):
        device = self.device
        self.predicted = torch.zeros(size=self.output_size, device=device)
        if self.with_target:
            self.target = torch.zeros(size=self.output_size, device=self.device)

    def compute(self):
        pass


class R2Metric(Metric):
    def __init__(self, config, loader) -> None:
        super(R2Metric, self).__init__("r2", config, loader)

    def compute(self):
        # Compute the mean of the target values
        target_mean = torch.mean(self.target, dim=0)
        print(self.target.shape)

        # Compute the total sum of squares (SS_tot)
        ss_tot = torch.sum((self.target - target_mean) ** 2, dim=0)
        print(ss_tot)

        # Compute the residual sum of squares (SS_res)
        ss_res = torch.sum((self.target - self.predicted) ** 2, dim=0)
        print(ss_res)

        print(ss_res / ss_tot)
        # Compute the R² score
        r2_score = 1 - ss_res / ss_tot
        print("r2_score", r2_score)

        # Return the mean R² score across all output dimensions
        return {self.name: r2_score}


class MSMetric(Metric):
    def __init__(self, config, loader) -> None:
        super(MSMetric, self).__init__("mean_square", config, loader)

    def _compute(self):
        diff = self.predicted - self.target
        return torch.mean(diff * diff, dim=0)

    def compute(self):
        ms = self.compute()
        self.results = {self.name: ms}
        return self.results


class RMSMetric(MSMetric):
    def __init__(self, config, loader) -> None:
        super(RMSMetric, self).__init__(config, loader)
        self.name = "rms"

    def compute(self):
        mean_squared_error = super()._compute()
        self.results = {self.name: torch.sqrt(mean_squared_error)}
        return self.results


class MetricError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MetricFactory(metaclass=MetaSingleton):
    def __init__(self) -> None:
        self.metrics_dict = {"r2": R2Metric, "rms": RMSMetric}

    def register_metric(self, key, metric_cls):
        self.metrics_dict[key] = metric_cls

    def create_metric(self, kind: str, config: dict, loader, device="cpu"):
        """Creates an architecture of NN

        Args:
            kind (str): kind of NN, key for the factory
            config (dict): config dict to use to initialize metric
            device (str, optional): Defaults to "cpu".

        Raises:
            MetricError: error raised if no architecture fit kind key

        Returns:
            iapytoo.metrics.Metric: metric
        """
        try:
            metric: Metric = self.metrics_dict[kind](config, loader)
        except KeyError:
            raise MetricError(f"metric {kind} is not handled")

        metric = metric.to(device=device)
        return metric


class MetricsCollection(Metric):
    def __init__(self, tag: str, metric_names: list, config, loader):
        super().__init__(tag, config, loader)
        self.metrics = {}
        factory = MetricFactory()
        try:
            for n in metric_names:
                self.metrics[f"{tag}_{n}"] = factory.create_metric(n, config, loader)
        except MetricError as er:
            print(f"Unable to create metric : {er}")

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
        for k, m in self.metrics.items():
            results = m.compute()
            self.results.update(results)
        return self.results

    # overwrite
    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()
