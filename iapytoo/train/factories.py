import torch.nn as nn
from iapytoo.utils.singleton import singleton
import torch.optim as to
from torch.optim.lr_scheduler import StepLR
from iapytoo.predictions.predictors import Predictor, MaxPredictor
from iapytoo.train.valuator import ModelValuator, WGANValuator
from iapytoo.metrics.metric import Metric
from iapytoo.metrics.predefined import R2Metric, RMSMetric, AccuracyMetric, MSMetric

from iapytoo.utils.config import Config, TrainingConfig


# Model

class WeightInitiator:
    def __call__(self, m):
        """method to be overloaded for weight initialization by use of 'model.apply(initiator)'"""
        classname = m.__class__.__name__
        print(f"init {classname}")


class Model(nn.Module):
    def __init__(self, loader, config) -> None:
        super().__init__()

    def weight_initiator(self):
        """return an WeightInitiator subclass that will be used to initialize weights of this model"""
        return None

    def predict(self, model_output):
        """This method gives the opportunity to transform output of the model to something equivalent to Y given by the dataset
        For example, class from probability distribution"""
        return model_output


# Optimizer

class Optimizer:
    def __init__(self, model, config: Config) -> None:
        self.torch_optimizer = None


class AdamOptimizer(Optimizer):
    def __init__(self, model, config: Config) -> None:
        super().__init__(model, config)

        training_config: TrainingConfig = config.training

        kwargs = {"lr": training_config.learning_rate}
        if training_config.weight_decay is not None:
            kwargs["weight_decay"] = training_config.weight_decay
        if training_config.betas is not None:
            kwargs["betas"] = training_config.betas

        self.torch_optimizer = to.Adam(model.parameters(), **kwargs)


class RMSpropOptimizer(Optimizer):
    def __init__(self, model, config: Config):
        super().__init__(model, config)
        self.torch_optimizer = to.RMSprop(
            model.parameters(), lr=config.training.learning_rate
        )


class SGDOptimizer(Optimizer):
    def __init__(self, model, config: Config) -> None:
        super().__init__(model, config)
        kwargs = {"lr": config.training.learning_rate}
        if config.training.weight_decay is not None:
            kwargs["weight_decay"] = config.training.weight_decay
        if config.training.momentum is not None:
            kwargs["momentum"] = config.training.momentum
        self.torch_optimizer = to.SGD(model.parameters(), **kwargs)

# Scheduler


class Scheduler:
    def __init__(self, optimizer, config: Config) -> None:
        self.lr_scheduler = None


class StepScheduler(Scheduler):
    def __init__(self, optimizer, config) -> None:
        super().__init__(optimizer, config)

        kwargs = {}
        kwargs["gamma"] = config.training.gamma
        kwargs["step_size"] = config.training.step_size
        self.lr_scheduler = StepLR(optimizer, **kwargs)

# Factories


class OptimizerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ModelError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class SchedulerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MetricError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


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

    def register_loss(self, key, loss_cls):
        self.loss_dict[key] = loss_cls

    def create_loss(self, kind: str):
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

    def create_scheduler(self, kind: str, optimizer, config: Config):
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

    def create_optimizer(self, kind: str, model, config: Config):
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

    def create_model(self, kind: str, config: Config, loader, device="cpu"):
        """Creates an architecture of NN

        Args:
            kind (str): kind of NN, key for the factory
            config (dict): config dict to use to initialize model
            loader (DataLoader): a dataloader to have input/output dimensions
            device (str, optional): Defaults to "cpu".

        Raises:
            ModelError: error raised if no architecture fit kind key

        Returns:
            nn.Module: pytorch model
        """
        try:
            model: Model = self.models_dict[kind](loader, config)
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

    def create_predictor(self, key: Config | str):
        if isinstance(key, Config):
            config: Config = key
            kind = config.model.predictor
        else:
            kind = key
        assert kind is not None, "no default predictor defined ?"

        return self.predictor_dict[kind]()

    def create_valuator(self, key: Config | str, model, device):
        if isinstance(key, Config):
            config: Config = key
            kind = config.model.valuator
        else:
            kind = key
        assert kind is not None, "no default valuator defined ?"

        return self.valuator_dict[kind](model, device)
