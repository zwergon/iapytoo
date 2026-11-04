import torch
import torch.nn as nn

from iapytoo.utils.config import Config
from torch.utils.data import DataLoader
from iapytoo.utils.model_config import DDPMConfig


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
        model_config = config.model
        if model_config.backbone:
            # Import differe : evite le cycle train/model.py -> train/factories.py ->
            # mlflow/model.py -> train/model.py (Factory importe MlflowModelProvider,
            # qui type-hint Model).
            from iapytoo.train.factories import Factory
            self.backbone = Factory().create_backbone(model_config.backbone, config)
        else:
            self.backbone = None

    def forward(self, *args, **kwargs):
        """Delegue au backbone (config.model.backbone) si defini. Les sous-classes
        qui ont leur propre architecture (pas de backbone pluggable) overrident
        forward() normalement et ne sont pas affectees."""
        assert self.backbone is not None, (
            "Model.forward() necessite config.model.backbone (ou l'override de forward())"
        )
        return self.backbone(*args, **kwargs)

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
    """
    Base des generateurs WGAN. Le bruit d'entree vient de get_noise() (a
    surcharger si le generateur attend autre chose qu'un vecteur gaussien de
    taille config.model.noise_dim), jamais d'un loader : les loaders d'un WGAN
    contiennent des donnees reelles, comme pour tout autre entrainement.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.noise_dim = getattr(config.model, "noise_dim", None)

    def get_noise(self, n_samples, noise_dim=None, device="cpu"):
        """Bruit d'entree du generateur : (n_samples, noise_dim) gaussien."""
        noise_dim = noise_dim if noise_dim is not None else self.noise_dim
        return torch.randn(n_samples, noise_dim, device=device)

    # override
    def evaluate_loader(self, loader: DataLoader):
        """
        Pour chaque batch reel du loader, genere autant d'echantillons a
        partir de get_noise() : renvoie (generes, reels) -- les plotters
        disposent ainsi des deux (PredictionType.PREDICTED / ACTUAL).
        """
        # self is generator
        device = self.device

        self.eval()
        with torch.no_grad():
            for batch in loader:
                X = batch[0] if isinstance(batch, (list, tuple)) else batch
                noise = self.get_noise(X.shape[0], self.noise_dim, device=device)
                outputs = self(noise).detach().cpu()

                yield outputs, X.detach().cpu()


class DDPMModel(Model):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        ddpm_config: DDPMConfig = config.model
        self.T = ddpm_config.n_times
        self.betas = torch.linspace(ddpm_config.beta_start, ddpm_config.beta_end, self.T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.t = None
        self.noise = None
        self.n_eval = getattr(ddpm_config, "n_eval", None)

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

    # override
    def evaluate_loader(self, loader: DataLoader):
        """
        Comme WGANModel.evaluate_loader : pour chaque batch reel du loader,
        genere autant d'echantillons (evaluate_one, depuis du bruit gaussien
        de meme forme) et renvoie (generes, reels) -- les plotters disposent
        ainsi des deux (PredictionType.PREDICTED / ACTUAL). S'arrete apres
        self.n_eval echantillons (config.model.n_eval, None = tout le loader).
        """
        device = self.device
        n_done = 0
        for batch in loader:
            X = batch[0] if isinstance(batch, (list, tuple)) else batch
            if self.n_eval is not None:
                if n_done >= self.n_eval:
                    break
                X = X[:self.n_eval - n_done]
            noise = torch.randn(X.shape, device=device)
            outputs = self.evaluate_one(noise).detach().cpu()
            n_done += X.shape[0]

            yield outputs, X.detach().cpu()

    def evaluate_one(self, x):
        self.eval()
        with torch.no_grad():
            for t in reversed(range(self.T)):
                z = torch.randn_like(x, device=x.device) if t > 0 else 0
                t_batch = torch.full((x.shape[0],), t, device=x.device)
                eps = self(x, t_batch.float()/self.T)
                x = (x - (1-self.alphas[t])/torch.sqrt(1-self.alphas_cumprod[t])
                     * eps)/torch.sqrt(self.alphas[t]) + torch.sqrt(self.betas[t])*z
        return x
