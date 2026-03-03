from iapytoo.utils.singleton import singleton

from iapytoo.utils.config import Config


from iapytoo.train.optimizer import (
    Optimizer,
    AdamOptimizer,
    SGDOptimizer,
    RMSpropOptimizer,
    OptimizerError,
)


from iapytoo.metrics.metric import (
    Metric,
    Metrics,
    MetricError
)

from iapytoo.metrics.predefined import (
    R2Metric,
    RMSMetric,
    AccuracyMetric,
    MSMetric,
    AccumulAccuracyMetric
)

from iapytoo.train.scheduler import (
    Scheduler,
    StepScheduler,
    SchedulerError
)

from iapytoo.mlflow.model import (
    ProviderError,
    MlflowModelProvider
)

from iapytoo.train.nn_loss import (
    NNLoss,
    MSELoss,
    NLLLoss,
    CrossEntropyLoss
)

from iapytoo.predictions.predictors import Predictor


@singleton
class Factory:
    def __init__(self) -> None:
        self.loss_dict = {
            "mse": MSELoss,
            "nll": NLLLoss,
            "cel": CrossEntropyLoss,
        }

        self.schedulers_dict = {
            "step": StepScheduler
        }

        self.optimizers_dict = {
            "adam": AdamOptimizer,
            "sgd": SGDOptimizer,
            "rmsprop": RMSpropOptimizer,
        }

        self.provider_dict = {
        }

        self.metrics_dict = {
            "r2": R2Metric,
            "rms": RMSMetric,
            "ms": MSMetric,
            "accuracy": AccuracyMetric,
            "accumul_accuracy": AccumulAccuracyMetric
        }

    def register_provider(self, provider_cls):
        self.provider_dict[provider_cls.__name__] = provider_cls

    def create_provider(self, kind: str, config: Config) -> MlflowModelProvider:
        try:
            provider = self.provider_dict[kind](config)
        except KeyError:
            raise ProviderError(f"transform {kind} is not handled")

        return provider

    def register_loss(self, key, loss_cls):
        self.loss_dict[key] = loss_cls

    def create_loss(self, kind: str, config: Config) -> NNLoss:
        """Creates a loss criterion

        Args:
            kind (str): kind of loss, key for the factory

        Returns:
            nn.Module: pytorch loss function
        """
        try:
            criterion = self.loss_dict[kind](config)
        except KeyError:
            raise Exception(f"loss {kind} is not handled")

        return criterion

    def register_scheduler(self, key, scheduler_cls):
        self.schedulers_dict[key] = scheduler_cls

    def create_scheduler(self, kind: str, optimizer, config: Config) -> Scheduler:
        """Creates an learning rate scheduler

        Args:
            kind (str): kind of scheduler, key for the factory
            config (dict): config dict to use to initialize scheduler

        Raises:
            SchedulerError: error raised if no architecture fit kind key

        Returns:
            nn.Module: pytorch scheduler
        """
        try:
            scheduler = self.schedulers_dict[kind](optimizer, config)
        except KeyError:
            raise SchedulerError(f"scheduler {kind} is not handled")

        return scheduler

    def register_optimizer(self, key, optimizer_cls):
        self.optimizers_dict[key] = optimizer_cls

    def create_optimizer(self, kind: str, model, config: Config) -> Optimizer:
        """Creates an optimizer for the NN

        Args:
            kind (str): kind of optimizer, key for the factory
            config (dict): config dict to use to initialize optimizer

        Raises:
            OptimizerError: error raised if no architecture fit kind key

        Returns:
            nn.Module: pytorch optimizer
        """
        try:
            optimizer = self.optimizers_dict[kind](model, config)
        except KeyError:
            raise OptimizerError(f"optimizer {kind} is not handled")

        return optimizer

    def register_metric(self, key, metric_cls):
        self.metrics_dict[key] = metric_cls

    def create_metrics(
        self,
        tag: str,
        kind: list[str],
        config: Config,
        predictor: Predictor | None = None,
        device: str = "cpu"
    ) -> Metrics:

        metric_list = []
        for k in kind:
            try:
                metric = self.metrics_dict[k](config)
            except KeyError:
                raise MetricError(f"Metric '{k}' is not registered")
            metric_list.append(metric)

        metrics = Metrics(tag, metric_list, config, predictor=predictor)
        metrics.to(device)

        return metrics
