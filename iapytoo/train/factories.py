import torch.nn as nn
from iapytoo.utils.meta_singleton import MetaSingleton
import torch.optim as to
from torch.optim.lr_scheduler import StepLR


class ModelError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


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

    def predict(self, X):
        return self.forward(X)


class ModelFactory(metaclass=MetaSingleton):
    def __init__(self) -> None:
        self.models_dict = {}

    def register_model(self, key, model_cls):
        self.models_dict[key] = model_cls

    def create_model(self, kind: str, config: dict, loader, device="cpu"):
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


class OptimizerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Optimizer:
    def __init__(self, model, config) -> None:
        self.torch_optimizer = None


class AdamOptimizer(Optimizer):
    def __init__(self, model, config) -> None:
        super().__init__(model, config)

        kwargs = {"lr": config["learning_rate"]}
        if "weight_decay" in config:
            kwargs["weight_decay"] = config["weight_decay"]
        if "betas" in config:
            kwargs["betas"] = config["betas"]

        self.torch_optimizer = to.Adam(model.parameters(), **kwargs)


class RMSpropOptimizer(Optimizer):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.torch_optimizer = to.RMSprop(
            model.parameters(), lr=config["learning_rate"]
        )


class SGDOptimizer(Optimizer):
    def __init__(self, model, config) -> None:
        super().__init__(model, config)
        self.torch_optimizer = to.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )


class OptimizerFactory(metaclass=MetaSingleton):
    def __init__(self) -> None:
        self.optimizers_dict = {
            "adam": AdamOptimizer,
            "sgd": SGDOptimizer,
            "rmsprop": RMSpropOptimizer,
        }

    def register_optimizer(self, key, optimizer_cls):
        self.optimizers_dict[key] = optimizer_cls

    def create_optimizer(self, kind: str, model, config: dict):
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


class SchedulerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Scheduler:
    def __init__(self, optimizer, config) -> None:
        self.lr_scheduler = None


class StepScheduler(Scheduler):
    def __init__(self, optimizer, config) -> None:
        super().__init__(optimizer, config)
        self.lr_scheduler = StepLR(optimizer, step_size=1, gamma=config["gamma"])


class SchedulerFactory(metaclass=MetaSingleton):
    def __init__(self) -> None:
        self.schedulers_dict = {"step": StepScheduler}

    def register_scheduler(self, key, scheduler_cls):
        self.schedulers_dict[key] = scheduler_cls

    def create_scheduler(self, kind: str, optimizer, config: dict):
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


class LossFactory(metaclass=MetaSingleton):
    def __init__(self) -> None:
        self.loss_dict = {"mse": nn.MSELoss, "nll": nn.NLLLoss}

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
