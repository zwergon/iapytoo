import torch.nn as nn
from iapytoo.utils.singleton import singleton

from iapytoo.utils.config import Config

from iapytoo.train.loss import Loss

from iapytoo.train.optimizer import (
    Optimizer,
    AdamOptimizer,
    SGDOptimizer,
    RMSpropOptimizer,
    OptimizerError,
)

from iapytoo.train.model import (
    Model,
    ModelError
)

from iapytoo.metrics.metric import (
    Metric,
    MetricError
)

from iapytoo.metrics.predefined import (
    R2Metric,
    RMSMetric,
    AccuracyMetric,
    MSMetric
)
from iapytoo.train.scheduler import (
    Scheduler,
    StepScheduler,
    SchedulerError
)

from iapytoo.train.mlflow_model import (
    ProviderError,
    MlflowModelProvider
)

from iapytoo.predictions.predictors import Predictor


@singleton
class Factory:
    def __init__(self) -> None:
        self.loss_dict = {
            "mse": nn.MSELoss,
            "nll": nn.NLLLoss,
            "cel": nn.CrossEntropyLoss,
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

    def create_loss(self, kind: str) -> Loss:
        """Creates a loss criterion

        Args:
            kind (str): kind of loss, key for the factory

        Returns:
            nn.Module: pytorch loss function
        """
        try:
            criterion = self.loss_dict[kind]()
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

    def create_metric(self, kind: str, config: Config, device="cpu", predictor: Predictor = None) -> Metric:
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
            metric: Metric = self.metrics_dict[kind](config, predictor)
        except KeyError:
            raise MetricError(f"metric {kind} is not handled")

        metric = metric.to(device=device)
        return metric
