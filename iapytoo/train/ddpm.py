
import torch
from enum import Enum

import torch.nn.functional as F

from iapytoo.train.training import Training
from iapytoo.utils.config import Config
from iapytoo.train.loss import Loss


from iapytoo.train.training import Training
from iapytoo.train.mlflow_model import save_mlflow_model
from iapytoo.utils.config import Config
from iapytoo.train.loss import Loss
from iapytoo.train.logger import Logger
from iapytoo.train.checkpoint import CheckPoint


class DDPM_LOSS(str, Enum):
    NOISE = 'noise'
    MODEL = 'model'


class DDPM(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.loss = Loss(DDPM_LOSS)
        self.T = 1000
        self.Tsig = 512
        self.betas = self.linear_beta_schedule()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self._lambda = 0.1

    def _create_criterion(self):
        return super()._create_criterion()

    def linear_beta_schedule(self):
        return torch.linspace(1e-4, 0.02, self.T)

    def q_sample(self, x0, t, noise):
        return (
            torch.sqrt(self.alphas_cumprod[t])[:, None, None]*x0 +
            torch.sqrt(1-self.alphas_cumprod[t])[:, None, None]*noise
        )

    # override
    def _inner_train(self, batch, batch_idx, d_metrics):

        real_data = batch
        real_data = real_data.to(self.device)

        t = torch.randint(0, self.T, (real_data.size(0),), device=self.device)

        noise = torch.randn_like(real_data)

        xt = self.q_sample(real_data, t, noise)

        pred_noise = self.model(xt, t.float()/self.T)
        loss_noise = F.mse_loss(pred_noise, noise)

        x0_hat = (xt - torch.sqrt(1-self.alphas_cumprod[t])[
                  :, None, None]*pred_noise) / torch.sqrt(self.alphas_cumprod[t])[:, None, None]

        loss_target = self.criterion(x0_hat, real_data)

        loss = loss_noise + self._lambda*loss_target
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )  # Clip gradient
        self.optimizer.step()

        losses = {
            "loss_noise": float(loss_noise),
            "loss_target": float(loss_target)
        }

        return losses

    # override
    def _train(self, epoch, train_loader):
        # Train
        self.model.train()
        return self.train_loop(epoch, train_loader, "Train")
