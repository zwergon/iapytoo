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

from iapytoo.predictions.predictors import (
    Predictor,
    MaxPredictor
)

from iapytoo.train.valuator import (
    Valuator,
    ModelValuator,
    WGANValuator
)

from iapytoo.dataset.transform import Transform, TransformError
from iapytoo.dataset.scaling import (
    MeanNormalize,
    MinMaxNormalize,
    MeanScalingByColumn,
    MinMaxScalingByColumn
)


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

        self.models_dict = {
        }

        self.metrics_dict = {
            "r2": R2Metric,
            "rms": RMSMetric,
            "ms": MSMetric,
            "accuracy": AccuracyMetric,
        }

        self.predictor_dict = {
            "default": Predictor,
            "max": MaxPredictor
        }

        self.valuator_dict = {
            "model": ModelValuator,
            "gan": WGANValuator
        }

        self.transform_dict = {
            "normalize": MeanNormalize,
            "minmax": MinMaxNormalize,
            "normalize_by_column": MeanScalingByColumn,
            "minmax_by_column": MinMaxScalingByColumn
        }

    def register_transform(self, key, transform_cls):
        self.transform_dict[key] = transform_cls

    def create_transform(self, kind: str, config: Config) -> Transform:
        try:
            transform = self.transform_dict[kind](config)
        except KeyError:
            raise TransformError(f"transform {kind} is not handled")

        return transform

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

    def register_model(self, key, model_cls):
        self.models_dict[key] = model_cls

    def create_model(self, kind: str, config: Config, device="cpu") -> Model:
        """Creates an architecture of NN

        Args:
            kind (str): kind of NN, key for the factory
            config (dict): config dict to use to initialize model
            device (str, optional): Defaults to "cpu".

        Raises:
            ModelError: error raised if no architecture fit kind key

        Returns:
            nn.Module: pytorch model
        """
        try:
            model: Model = self.models_dict[kind](config)
        except KeyError:
            raise ModelError(f"model {kind} is not handled")

        initiator = model.weight_initiator()
        if initiator is not None:
            model.apply(initiator)

        model = model.to(device=device)
        return model

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

    def register_predictor(self, key, predictor_cls):
        self.predictor_dict[key] = predictor_cls

    def create_predictor(self, key: Config | str, *args, **kwargs) -> Predictor:
        if isinstance(key, Config):
            config: Config = key
            kind = config.model.predictor
        else:
            kind = key
        assert kind is not None, "no default predictor defined ?"

        return self.predictor_dict[kind](*args, **kwargs)

    def create_valuator(self, key: Config | str, model, device) -> Valuator:
        if isinstance(key, Config):
            config: Config = key
            kind = config.model.valuator
        else:
            kind = key
        assert kind is not None, "no default valuator defined ?"

        return self.valuator_dict[kind](model, device)
