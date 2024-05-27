import torch.nn as nn
from iapytoo.utils.meta_singleton import MetaSingleton


class ModelError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Model(nn.Module):
    def __init__(self, loader, config) -> None:
        super().__init__()

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
            model = self.models_dict[kind](loader, config)
        except KeyError:
            raise ModelError(f"model {kind} is not handled")

        model = model.to(device=device)
        return model
