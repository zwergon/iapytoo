import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


from iapytoo.train.wgan import WGAN
from iapytoo.utils.config import Config
from iapytoo.train.factories import Model, ModelFactory
from iapytoo.predictions.plotters import Fake2DPlotter
from iapytoo.train.factories import WeightInitiator


class MNISTInitiator(WeightInitiator):
    def __call__(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)


class Generator(Model):
    """See https://github.com/Zeleni9/pytorch-wgan"""

    @staticmethod
    def get_noise(n_samples, z_dim, device="cpu"):
        """
        Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
        creates a tensor of that shape filled with random numbers from the normal distribution.
        Parameters:
            n_samples: the number of samples to generate, a scalar
            z_dim: the dimension of the noise vector, a scalar
            device: the device type
        """
        return torch.randn(n_samples, z_dim, device=device)

    def __init__(self, loader, config: Config):
        super(Generator, self).__init__(loader, config)
        self.z_dim = config.model.noise_dim
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = 1 (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(
                in_channels=self.z_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            # State (1024x4x4)
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            # State (512x8x8)
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            # State (256x16x16)
            nn.ConvTranspose2d(
                in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1
            ),
        )
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def unsqueeze_noise(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        return noise.view(noise.shape[0], self.z_dim, 1, 1)

    def forward(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        x = self.unsqueeze_noise(noise)
        x = self.main_module(x)
        return self.output(x)


class Discriminator(Model):
    def __init__(self, loader, config : Config):
        super(Discriminator, self).__init__(loader, config)
        # Filters [256, 512, 1024]
        # Input_dim = 1 (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(
                in_channels=1, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (256x16x16)
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State (512x8x8)
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1
            ),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(
                in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0
            )
        )

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)


class LatentDataset(Dataset):
    def __init__(self, noise_dim, size) -> None:
        super().__init__()
        self.noise = torch.randn(size, noise_dim)

    def __len__(self):
        return self.noise.shape[0]

    def __getitem__(self, idx):
        return self.noise[idx, :]


if __name__ == "__main__":
    config = Config.create_from_yaml(os.path.join(os.path.dirname(__file__), "config_wgan.yml"))

    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # load training data
    trainset = MNIST(config.dataset.path, download=True, train=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.dataset.batch_size, shuffle=True, drop_last=True
    )

    latentset = LatentDataset(config.model.noise_dim, size=16)

    valid_loader = torch.utils.data.DataLoader(latentset, batch_size=1, shuffle=False)

    item = next(iter(valid_loader))
    print(item.shape)

    model_factory = ModelFactory()
    model_factory.register_model("generator", Generator)
    model_factory.register_model("critic", Discriminator)

    wgan = WGAN(config)
    wgan.predictions.add_plotter(Fake2DPlotter())
    wgan.fit(train_loader=trainloader, valid_loader=valid_loader, run_id=None)
