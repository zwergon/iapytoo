import torch
from enum import IntEnum, Enum

from iapytoo.train.training import Training
from iapytoo.utils.config import Config
from iapytoo.mlflow.model import MlflowWGANProvider
from iapytoo.train.factories import Factory
from iapytoo.train.loss import Loss


class WGAN_FCT(IntEnum):
    GENERATOR = 0
    DISCRIMINATOR = 1


class WGAN_LOSS(str, Enum):
    GENERATOR = 'g_loss'
    DISCRIMINATOR = 'd_loss'
    # Distance de Wasserstein estimee E[D(reel)] - E[D(genere)] sur le
    # valid_loader (sans penalite de gradient) -- cf. WGAN._inner_validate.
    VALIDATE = 'w_dist_valid'


class WGAN(Training):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # one for generator, one for discriminator
        self.loss = Loss(WGAN_LOSS)

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

    def _create_models(self):
        assert self.mlflow_model_provider is not None, "generator is defined as model in provider"

        gan_provider: MlflowWGANProvider = self.mlflow_model_provider

        generator = gan_provider.generator
        generator.to(self.device)

        discriminator = gan_provider.discriminator
        discriminator.to(self.device)

        return [generator, discriminator]

    def _create_optimizers(self):
        factory = Factory()
        g_optimizer = factory.create_optimizer(
            self._config.training.optimizer, self.generator, self._config
        )
        d_optimizer = factory.create_optimizer(
            self._config.training.optimizer, self.discriminator, self._config
        )

        return [g_optimizer, d_optimizer]

    def _create_metrics(self):
        return {
            phase: Factory().create_metrics(
                phase,
                self._config.metrics.names,
                self._config,
                predictor=self.predictor,
                device=self.device
            )
            for phase in ("Train", "Valid")
        }

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

    def _update_d(self, real_data, real_labels=None):
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

        self.loss(WGAN_LOSS.DISCRIMINATOR).update(d_loss.item())
        return d_loss.item()

    def _update_g(self, real_data, real_labels=None):
        noise_dim = self._config.model.noise_dim
        # Taille du batch reel (et non config.dataset.batch_size) : correct
        # aussi pour un dernier batch incomplet (DataLoader sans drop_last).
        batch_size = real_data.shape[0]

        # Mise à jour du générateur
        noise = self.generator.get_noise(
            batch_size, noise_dim, device=self.device)
        fake_data = self.generator(noise)

        g_loss = -self.discriminator(fake_data).mean()
        g_loss.backward()
        self.g_optimizer.step()

        self.loss(WGAN_LOSS.GENERATOR).update(g_loss.item())
        return g_loss.item()

    # Pas d'override de _on_epoch_ended : celui de Training (checkpoint tous
    # les config.checkpoint_epoch, log du learning rate, predictions seulement
    # si des plotters sont definis ET qu'un loader est fourni) s'applique tel
    # quel. L'ancien override appelait predictions.compute(loader=None) des
    # l'epoch 0 quand fit() etait appele sans valid_loader (fit() passe
    # toujours loader=valid_loader, meme None), et ne sauvegardait jamais de
    # checkpoint (reprise d'entrainement impossible).

    @staticmethod
    def _split_batch(batch):
        """batch = X seul (tenseur) ou (X, Y) / (X,) : renvoie (X, Y ou None)."""
        if isinstance(batch, (list, tuple)):
            return batch[0], (batch[1] if len(batch) > 1 else None)
        return batch, None

    # override
    def _inner_train(self, batch, batch_idx):
        # Un dataset generatif n'a en general pas de cible : X seul accepte.
        real_data, real_labels = self._split_batch(batch)
        real_data = real_data.to(self.device)
        if real_labels is not None:
            real_labels = real_labels.to(self.device)

        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

        d_loss = self._update_d(real_data, real_labels)

        n_critic = self._config.model.n_critic
        if n_critic == 1 or batch_idx % n_critic == 0:
            g_loss = self._update_g(real_data, real_labels)
        else:
            g_loss = None

        losses = {
            "d_loss": float(d_loss)
        }
        if g_loss is not None:
            losses["g_loss"] = float(g_loss)

        return losses

    # override
    def _train(self, epoch, train_loader):
        # Train
        self.generator.train()
        self.discriminator.train()
        return self.train_loop(epoch, train_loader, "Train")

    # override
    def _inner_validate(self, batch, batch_idx):
        """
        valid_loader = donnees reelles (comme pour tout Training ; le bruit du
        generateur vient de get_noise(), cf. WGANModel). Estime la distance de
        Wasserstein E[D(reel)] - E[D(genere)] sur le batch, sans penalite de
        gradient : l'indicateur de convergence usuel d'un WGAN (d_loss inclut
        la penalite, g_loss depend du niveau arbitraire du critique). Pilote
        aussi le scheduler (ex. plateau), comme la valid_loss de Training.
        """
        real_data, _ = self._split_batch(batch)
        real_data = real_data.to(self.device)
        noise = self.generator.get_noise(
            real_data.shape[0], self._config.model.noise_dim, device=self.device)

        w_dist = (self.discriminator(real_data).mean()
                  - self.discriminator(self.generator(noise)).mean()).item()

        if self.scheduler is not None:
            self.scheduler.update(w_dist)

        self.loss(WGAN_LOSS.VALIDATE).update(w_dist)
        return {WGAN_LOSS.VALIDATE.value: w_dist}

    # override
    def _validate(self, epoch, valid_loader):
        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            return self.valid_loop(epoch, valid_loader, "Valid")
