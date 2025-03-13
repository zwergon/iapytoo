import sys
import torch
from torch.autograd import grad
from tqdm import tqdm
import logging

from iapytoo.dataset.scaling import Scaling
from iapytoo.predictions import GenerativePredictions, Predictor
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
        predictor: Predictor = Predictor(),
        metric_creators: list = ...,
        y_scaling: Scaling = None,
    ) -> None:
        super().__init__(config, predictor, metric_creators, y_scaling)
        # one for generator, one for discriminator
        self.loss = Loss(n_losses=2)
        if self._config.training.tqdm:
            self.train_loop = self.__tqdm_gan_loop(
                self._update_g, self._update_d)
        else:
            self.train_loop = self.__batch_gan_loop(
                self._update_g, self._update_d)
        self.predictions = GenerativePredictions(predictor=predictor)

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
            self._config.model.generator, self._config, loader, self.device
        )
        discriminator = ModelFactory().create_model(
            self._config.model.discriminator, self._config, loader, self.device
        )

        return [generator, discriminator]

    def _create_optimizers(self):
        g_optimizer = OptimizerFactory().create_optimizer(
            self._config.training.optimizer, self.generator, self._config
        )
        d_optimizer = OptimizerFactory().create_optimizer(
            self._config.training.optimizer, self.discriminator, self._config
        )

        return [g_optimizer, d_optimizer]

    @staticmethod
    def freeze_params(model, freeze: bool):
        for param in model.parameters():
            param.requires_grad = freeze

    # Fonction pour la pénalité de gradient
    def gradient_penalty(self, real, fake):
        # compute a shape compatible with real
        epsilon_shape = [1] * len(real.shape)
        epsilon_shape[0] = real.shape[0]

        # epsilon: a vector of the uniformly random proportions of real/fake per mixed image
        epsilon = torch.rand(
            epsilon_shape, device=real.device, requires_grad=True)

        # Mix the images together
        mixed_images = real * epsilon + fake * (1 - epsilon)

        # Calculate the critic's scores on the mixed images
        mixed_scores = self.discriminator(mixed_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            # Note: You need to take the gradient of outputs with respect to inputs.
            # This documentation may be useful, but it should not be necessary:
            # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
            inputs=mixed_images,
            outputs=mixed_scores,
            # These other parameters have to do with the pytorch autograd engine works
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Flatten the gradients so that each row captures one image
        gradient = gradient.view(len(gradient), -1)

        # Calculate the magnitude of every row
        gradient_norm = gradient.norm(2, dim=1)

        # Penalize the mean squared distance of the gradient norms from 1
        penalty = torch.mean((gradient_norm - 1) ** 2)

        return penalty

    def _update_d(self, real_data):
        noise_dim = self._config.model.noise_dim
        lambda_gp = self._config.model.lambda_gp

        noise = self.generator.get_noise(
            real_data.shape[0], noise_dim, device=self.device
        )
        with torch.no_grad():
            fake_data = self.generator(noise)

        # Pertes pour les vrais et faux
        d_fake = self.discriminator(fake_data)
        d_real = self.discriminator(real_data)

        # Pénalité de gradient
        gp = self.gradient_penalty(real_data, fake_data)

        # Perte totale du discriminateur
        d_loss = d_fake.mean() - d_real.mean() + lambda_gp * gp
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss.item()

    def _update_g(self):
        noise_dim = self._config.model.noise_dim
        batch_size = self._config.dataset.batch_size

        # Mise à jour du générateur
        noise = self.generator.get_noise(
            batch_size, noise_dim, device=self.device)
        fake_data = self.generator(noise)

        g_loss = -self.discriminator(fake_data).mean()
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item()

    def _on_epoch_ended(self, epoch, checkpoint, **kwargs):
        if epoch % 10 == 0:
            if "loader" in kwargs:
                self.predictions.compute(self, kwargs["loader"])
                self.logger.report_prediction(epoch, self.predictions)

        for item in self.loss(WGAN_FCT.GENERATOR).buffer:
            self.logger.report_metric(
                epoch=item[0], metrics={f"g_loss": item[1]})
        for item in self.loss(WGAN_FCT.DISCRIMINATOR).buffer:
            self.logger.report_metric(
                epoch=item[0], metrics={f"d_loss": item[1]})
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
                    real_data = real_data.to(self.device)
                    self.d_optimizer.zero_grad()
                    self.g_optimizer.zero_grad()

                    d_loss = update_d(real_data)
                    d_mean.update(d_loss)

                    # TODO add clip_value in Config.
                    # if "clip_value" in self.config:
                    #     clip_value = self.config["clip_value"]
                    #     for p in self.discriminator.parameters():
                    #         p.data.clamp_(-clip_value, clip_value)

                    # update generator not so often
                    n_critic = self._config.model.n_critic
                    if n_critic == 1 or step % n_critic == 0:
                        g_loss = update_g()
                        g_mean.update(g_loss)

                    timer.tick()

                    tepoch.set_postfix(d_loss=d_mean.value,
                                       g_loss=g_mean.value)

            timer.log()
            timer.stop()

        return new_function

    def __batch_gan_loop(self, update_g, update_d):
        """
        This is a decorator that encapsulates the inner learning procces.
        Iterations over all batches of one epoch.
        This decorator displays a progress bar and computes some times
        """

        def new_function(epoch, loader, description):
            timer = Timer()
            timer.start()
            d_mean = self.loss(WGAN_FCT.DISCRIMINATOR)
            g_mean = self.loss(WGAN_FCT.GENERATOR)

            logging.info(
                f"Epoch {epoch} {description}"
            )

            size_by_batch = len(loader)
            step = max(size_by_batch //
                       self._config.training.n_steps_by_batch, 1)
            for i, (real_data, _) in enumerate(loader):

                if i % step == 0:
                    f"Processing batch {i+1}/{size_by_batch}"

                # update discriminator
                real_data = real_data.to(self.device)
                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                d_loss = update_d(real_data)
                d_mean.update(d_loss)

                # TODO add clip_value in Config.
                # if "clip_value" in self.config:
                #     clip_value = self.config["clip_value"]
                #     for p in self.discriminator.parameters():
                #         p.data.clamp_(-clip_value, clip_value)

                # update generator not so often
                n_critic = self._config.model.n_critic
                if n_critic == 1 or i % n_critic == 0:
                    g_loss = update_g()
                    g_mean.update(g_loss)

                timer.tick()

                if i % step == 0:
                    logging.info(
                        f"Step {i+1}/{size_by_batch} - d_loss: {d_mean.value:.6f}, g_loss: {g_mean.value:.6f}")

            timer.log()
            timer.stop()

        return new_function

    def __train(self, epoch, train_loader):
        # Train
        self.generator.train()
        self.discriminator.train()
        return self.train_loop(epoch, train_loader, "Train")

    def fit(self, train_loader, valid_loader, run_id=None):
        num_epochs = self._config.training.epochs

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

                self._on_epoch_ended(epoch, checkpoint, loader=valid_loader)

            self.logger.save_model(self.model)

        return {
            "run_id": self.logger.run_id,
            "run_name": active_run_name,
            "g_loss": self.loss(WGAN_FCT.GENERATOR).value,
            "d_loss": self.loss(WGAN_FCT.DISCRIMINATOR).value,
        }
