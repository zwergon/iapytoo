import torch
import torch.nn as nn
from torch.types import Device

from iapytoo.utils.config import Config
from torch.utils.data import DataLoader
from iapytoo.utils.model_config import GanConfig, DDPMConfig


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

    # override
    def weight_initiator(self):
        """return an WeightInitiator subclass that will be used to initialize weights of this model"""
        pass

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

    def __init__(self, config: Config) -> None:
        super().__init__(config)

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


class DDPMModel(Model):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        ddpm_config: DDPMConfig = config.model
        self.T = ddpm_config.n_times
        self.betas = torch.linspace(1e-4, 0.02, self.T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.t = None
        self.noise = None

    def to(self, device):
        super().to(device)
        print(f"move to {device}")
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)

    @property
    def normalized_time(self):
        return self.t.float() / self.T

    def q_sample(self, real_data: torch.Tensor):
        noise = torch.randn_like(real_data, device=real_data.device)
        self.t = torch.randint(
            0, self.T, (real_data.shape[0],), device=real_data.device)
        return (
            torch.sqrt(self.alphas_cumprod[self.t])[:, None, None]*real_data +
            torch.sqrt(1-self.alphas_cumprod[self.t])[:, None, None]*noise,
            noise
        )

    def predict(self, xt: torch.Tensor, pred_noise: torch.Tensor):
        assert self.t is not None
        return (xt - torch.sqrt(1-self.alphas_cumprod[self.t])[
            :, None, None]*pred_noise) / torch.sqrt(self.alphas_cumprod[self.t])[:, None, None]

    def evaluate_loader(self, loader: DataLoader):
        return super().evaluate_loader(loader)

    def evaluate_one(self, x):
        for t in reversed(range(self.T)):
            z = torch.randn_like(x) if t > 0 else 0
            t_batch = torch.full((x.shape[0],), t)
            eps = self(x, t_batch.float()/self.T)
            x = (x - (1-self.alphas[t])/torch.sqrt(1-self.alphas_cumprod[t])
                 * eps)/torch.sqrt(self.alphas[t]) + torch.sqrt(self.betas[t])*z
        return x
