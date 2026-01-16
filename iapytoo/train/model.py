import torch
import torch.nn as nn


from iapytoo.utils.config import Config
from torch.utils.data import DataLoader


class ModelError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class WeightInitiator:
    def __call__(self, m):
        """method to be overloaded for weight initialization by use of 'model.apply(initiator)'"""
        classname = m.__class__.__name__
        print(f"init {classname}")


class Model(nn.Module):
    """
    The base class for all trainable models in iapytoo.

    Users must subclass this class to define:

    - the neural architecture
    - how raw model outputs are transformed into predictions

    Example:

    >>> class MyModel(Model):
    ...    def __init__(self, config: Config) -> None:
    ...        super().__init__(config)
    ...        # Define your layers here
    ...
    ...    def forward(self, x):
    ...        # Define the forward pass
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def weight_initiator(self):
        """return an WeightInitiator subclass that will be used to initialize weights of this model"""
        return None

    def predict(self, model_output: torch.Tensor) -> torch.Tensor:
        """Transform raw model outputs into prediction values.

        This method converts the output of the neural network into a format
        compatible with the dataset target (Y).

        Args: 
            model_output (torch.Tensor): Raw output of the neural network.

        Returns:
            torch.Tensor: Prediction value (e.g. class index, probability distribution).
        """
        return model_output

    def evaluate_one(self, input):

        self.eval()
        with torch.no_grad():
            return self(input)

    def evaluate_loader(self, loader: DataLoader):
        self.eval()
        with torch.no_grad():
            for X, Y in loader:
                X = X.to(self.device)
                model_output = self(X)
                outputs = model_output.detach().cpu()
                actual = Y.detach().cpu()
                yield outputs, actual


class WGANModel(Model):

    # override
    def evaluate_loader(self, loader: DataLoader):

        # self is generator
        device = self.device

        self.eval()
        with torch.no_grad():
            for X in loader:
                X = X.to(device)
                gen_output = self(X)

                outputs = gen_output.detach().cpu()

                yield outputs, None
