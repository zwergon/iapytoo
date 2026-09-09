
import torch
from enum import Enum

import torch.nn.functional as F

from iapytoo.train.training import Training
from iapytoo.utils.config import Config
from iapytoo.train.loss import Loss

from iapytoo.train.model import DDPMModel
from iapytoo.utils.model_config import DDPMConfig


class DDPM_LOSS(str, Enum):
    NOISE = 'noise'
    MODEL = 'model'
    VALIDATE = 'validate'


class DDPM(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.loss = Loss(DDPM_LOSS)
        ddpm_config: DDPMConfig = config.model
        self._lambda = ddpm_config.lambda_

    # override
    def _inner_train(self, batch, batch_idx):

        model: DDPMModel = self.model
        real_data = batch
        real_data = real_data.to(self.device)

        xt, noise = model.q_sample(real_data)

        pred_noise = model(xt, model.normalized_time)

        loss_noise = F.mse_loss(pred_noise, noise)
        self.loss(DDPM_LOSS.NOISE).update(loss_noise.item())

        x0_hat = model.predict(xt, pred_noise)

        loss_target = self.criterion(x0_hat, real_data)
        self.loss(DDPM_LOSS.MODEL).update(loss_target.item())

        try:
            self._metrics["Train"].update(x0_hat.detach(), real_data)
        except KeyError:
            pass

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
            DDPM_LOSS.NOISE: loss_noise.item(),
            DDPM_LOSS.MODEL: loss_target.item()
        }

        return losses

    # override
    def _inner_validate(self, batch, batch_idx):

        model: DDPMModel = self.model
        real_data = batch
        real_data = real_data.to(self.device)

        # Seed deterministe par batch_idx : le meme (t, bruit) est tire a chaque
        # epoch pour ce batch, donc seule la prediction du modele fait varier la
        # loss d'une epoch a l'autre (sinon q_sample tire un t/bruit different a
        # chaque appel et la loss de validation n'est plus comparable epoch a epoch).
        rng_state = torch.get_rng_state()
        torch.manual_seed(42 + batch_idx)

        xt, noise = model.q_sample(real_data)

        pred_noise = model(xt, model.normalized_time)

        torch.set_rng_state(rng_state)

        loss_noise = F.mse_loss(pred_noise, noise)

        x0_hat = model.predict(xt, pred_noise)

        loss_target = self.criterion(x0_hat, real_data)

        try:
            self._metrics["Valid"].update(x0_hat.detach(), real_data)
        except KeyError:
            pass

        loss = loss_noise + self._lambda*loss_target

        if self.scheduler is not None:
            self.scheduler.update(loss.item())

        self.loss(DDPM_LOSS.VALIDATE).update(loss.item())

        return {DDPM_LOSS.VALIDATE: loss.item()}

    # override
    def _train(self, epoch, train_loader):
        # Train
        self.model.train()
        return self.train_loop(epoch, train_loader, "Train")
