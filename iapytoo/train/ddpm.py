
import torch
from enum import Enum

import torch.nn.functional as F

from iapytoo.train.training import Training
from iapytoo.utils.config import Config
from iapytoo.train.loss import Loss


from iapytoo.train.training import Training
from iapytoo.utils.config import Config
from iapytoo.train.loss import Loss
from iapytoo.train.model import DDPMModel
from iapytoo.utils.model_config import DDPMConfig


class DDPM_LOSS(str, Enum):
    NOISE = 'noise'
    MODEL = 'model'


class DDPM(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.loss = Loss(DDPM_LOSS)
        ddpm_config: DDPMConfig = config.model
        self._lambda = ddpm_config.lambda_

    # override

    def _inner_train(self, batch, batch_idx, d_metrics):

        model: DDPMModel = self.model
        real_data = batch
        real_data = real_data.to(self.device)

        xt, noise = model.q_sample(real_data)

        pred_noise = model(xt, model.normalized_time)
        loss_noise = F.mse_loss(pred_noise, noise)

        x0_hat = model.predict(xt, pred_noise)

        loss_target = self.criterion(x0_hat, real_data)

        loss = loss_noise + self._lambda*loss_target
        self.optimizer.zero_grad()
        loss.backward()

        if self.scheduler is not None:
            self.scheduler.update(loss.item())

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
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
