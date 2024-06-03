import sys
import torch
from torch.autograd import grad
from tqdm import tqdm

from iapytoo.dataset.scaling import Scaling
from iapytoo.predictions import Predictions, PredictionPlotter
from iapytoo.train.training import Training
from iapytoo.utils.config import Config
from iapytoo.train.factories import (
    ModelFactory,
    OptimizerFactory,
    LossFactory,
)
from iapytoo.train.loss import Loss
from iapytoo.train.logger import Logger
from iapytoo.train.checkpoint import CheckPoint
from iapytoo.utils.timer import Timer
from enum import IntEnum


class WGAN_FCT(IntEnum):
    GENERATOR = 0
    DISCRIMINATOR = 1


class WGAN(Training):
    def __init__(
        self,
        config: Config,
        metric_creators: list = ...,
        prediction_plotter: PredictionPlotter = None,
        y_scaling: Scaling = None,
    ) -> None:
        super().__init__(config, metric_creators, prediction_plotter, y_scaling)
        self.loss = Loss(n_losses=2)  # one for generator, one for discriminator
        self.train_loop = self.__tqdm_gan_loop(self._update_g, self._update_d)

    # Fonction pour la pénalité de gradient
    def gradient_penalty(self, real_data, fake_data):
        b_size = real_data.size(0)
        alpha = torch.Tensor(b_size, 1, 1).uniform_(0, 1).to(real_data.device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_(True)

        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(
            d_interpolates.size(), requires_grad=False, device=real_data.device
        )

        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    @property
    def generator(self):
        return self._models[WGAN_FCT.GENERATOR]

    @property
    def discriminator(self):
        return self._models[WGAN_FCT.DISCRIMINATOR]

    @property
    def g_optimizer(self):
        return self._optimizers[WGAN_FCT.GENERATOR].torch_optimizer

    @property
    def d_optimizer(self):
        return self._optimizers[WGAN_FCT.DISCRIMINATOR].torch_optimizer

    def _create_criterion(self):
        return None

    def _create_models(self, loader):
        generator = ModelFactory().create_model(
            self.config["generator"], self.config, loader, self.device
        )
        discriminator = ModelFactory().create_model(
            self.config["discriminator"], self.config, loader, self.device
        )

        return [generator, discriminator]

    def _create_optimizers(self):
        g_optimizer = OptimizerFactory().create_optimizer(
            self.config["optimizer"], self.generator, self.config
        )
        d_optimizer = OptimizerFactory().create_optimizer(
            self.config["optimizer"], self.discriminator, self.config
        )

        return [g_optimizer, d_optimizer]

    def _update_d(self, real_data):
        noise_dim = self.config["noise_dim"]
        batch_size = self.config["batch_size"]
        lambda_gp = self.config["lambda_gp"]

        self.discriminator.zero_grad()

        noise = torch.randn(batch_size, noise_dim, 1, device=self.device)
        fake_data = self.generator(noise)

        real_data_reshaped = real_data.view(batch_size, 1, -1).to(self.device)

        # Pertes pour les vrais et faux
        d_real = self.discriminator(real_data_reshaped)
        d_fake = self.discriminator(fake_data)

        # Pénalité de gradient
        gp = self.gradient_penalty(real_data_reshaped, fake_data)

        # Perte totale du discriminateur
        d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def _update_g(self):
        noise_dim = self.config["noise_dim"]
        batch_size = self.config["batch_size"]

        # Mise à jour du générateur
        noise = torch.randn(batch_size, noise_dim, 1, device=self.device)

        self.generator.zero_grad()
        fake_data = self.generator(noise)

        g_loss = -self.discriminator(fake_data).mean()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()

    def _on_epoch_ended(self, epoch, checkpoint):
        for item in self.loss(WGAN_FCT.GENERATOR).buffer:
            self.logger.report_metric(epoch=item[0], metrics={f"g_loss": item[1]})
        for item in self.loss(WGAN_FCT.DISCRIMINATOR).buffer:
            self.logger.report_metric(epoch=item[0], metrics={f"d_loss": item[1]})
        self.loss.flush()

    def __tqdm_gan_loop(self, update_g, update_d):
        """
        This is a decorator that encapsulates the inner learning procces.
        Iterations over all batches of one epoch.
        This decorator displays a progress bar and computes some times
        """

        def new_function(epoch, loader, description):
            timer = Timer()
            timer.start()
            with tqdm(loader, unit="batch", file=sys.stdout) as tepoch:
                d_mean = self.loss(WGAN_FCT.DISCRIMINATOR)
                g_mean = self.loss(WGAN_FCT.GENERATOR)
                tepoch.set_description(f"{description} {epoch}")
                for step, (real_data, _) in enumerate(tepoch):
                    # update discriminator
                    d_loss = update_d(real_data)
                    d_mean.update(d_loss)

                    # update generator not so often
                    if step % 5:
                        g_loss = update_g()
                        g_mean.update(g_loss)

                    timer.tick()

                    tepoch.set_postfix(d_loss=d_mean.value, g_loss=g_mean.value)

            timer.log()
            timer.stop()

        return new_function

    def __train(self, epoch, train_loader):
        # Train
        return self.train_loop(epoch, train_loader, "Train")

    def fit(self, train_loader, valid_loader, run_id=None):
        num_epochs = self.config["epochs"]

        self.loss.reset()

        self._models = self._create_models(train_loader)
        self._optimizers = self._create_optimizers()

        checkpoint = CheckPoint(run_id)
        checkpoint.init(self)

        with Logger(self._config, run_id=checkpoint.run_id) as self.logger:
            active_run_name = self.logger.active_run_name()
            self._display_device()
            self.logger.set_signature(train_loader)
            self.logger.summary()

            for epoch in range(checkpoint.epoch + 1, num_epochs):
                # Train
                self.__train(epoch, train_loader)

                # increments scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self._on_epoch_ended(epoch, checkpoint)

            self.logger.save_model(self.model)

        return {
            "run_id": self.logger.run_id,
            "run_name": active_run_name,
            "g_loss": self.loss(WGAN_FCT.GENERATOR).value,
            "d_loss": self.loss(WGAN_FCT.DISCRIMINATOR).value,
        }
