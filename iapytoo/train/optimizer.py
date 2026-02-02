import torch.optim as to

from iapytoo.utils.config import Config, TrainingConfig


class OptimizerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


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
