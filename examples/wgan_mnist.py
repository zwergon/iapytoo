import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad

from iapytoo.train.wgan import WGAN
from iapytoo.utils.config import Config


from iapytoo.train.factories import Model, ModelFactory, OptimizerFactory
from iapytoo.predictions.plotters import PredictionPlotter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from iapytoo.train.factories import WeightInitiator


class MNISTInitiator(WeightInitiator):
    def __call__(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)


class Generator(Model):
    """
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, loader, config):
        super(Generator, self).__init__(loader, config)
        self.z_dim = config["noise_dim"]
        im_chan = 1
        hidden_dim = 64
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(self.z_dim, hidden_dim * 4),
            self.make_gen_block(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1
            ),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def weight_initiator(self):
        return MNISTInitiator()

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

    def make_gen_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        final_layer=False,
    ):
        """
        Function to return a sequence of operations corresponding to a generator block of DCGAN,
        corresponding to a transposed convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """

        #     Steps:
        #       1) Do a transposed convolution using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a ReLU activation.
        #       4) If its the final layer, use a Tanh activation after the deconvolution.

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.Tanh(),
            )

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
        return self.gen(x)


class Discriminator(Model):
    """
    Discriminator Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
    hidden_dim: the inner dimension, a scalar
    """

    def __init__(self, loader, config):
        super(Discriminator, self).__init__(loader, config)
        im_chan = 1
        hidden_dim = 16
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def weight_initiator(self):
        return MNISTInitiator()

    def make_disc_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        final_layer=False,
    ):
        """
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN,
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """
        #     Steps:
        #       1) Add a convolutional layer using the given parameters.
        #       2) Do a batchnorm, except for the last layer.
        #       3) Follow each batchnorm with a LeakyReLU activation with slope 0.2.
        #       Note: Don't use an activation on the final layer

        # Build the neural block
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:  # Final Layer
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    def forward(self, image):
        """
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


class LatentDataset(Dataset):
    def __init__(self, noise_dim, size) -> None:
        super().__init__()
        self.noise = torch.randn(size, noise_dim)

    def __len__(self):
        return self.noise.shape[0]

    def __getitem__(self, idx):
        return self.noise[idx, :]


class GridPlotter(PredictionPlotter):
    def __init__(self, n_plot=4):
        super().__init__()
        self.n_plot = n_plot

    def plot(self, epoch):
        fake_list = self.predictions.predicted
        grid_img = torchvision.utils.make_grid(fake_list, nrow=self.n_plot)

        f, ax = plt.subplots()
        ax.imshow(grid_img.permute(1, 2, 0))

        return "Generated", f


if __name__ == "__main__":
    config = Config(os.path.join(os.path.dirname(__file__), "config_wgan.json"))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # load training data
    trainset = MNIST(config.data_dir, download=True, train=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    latentset = LatentDataset(config.noise_dim, size=16)

    valid_loader = torch.utils.data.DataLoader(latentset, batch_size=1, shuffle=False)

    item = next(iter(valid_loader))
    print(item.shape)

    model_factory = ModelFactory()
    model_factory.register_model("generator", Generator)
    model_factory.register_model("critic", Discriminator)

    wgan = WGAN(config, prediction_plotter=GridPlotter())
    wgan.fit(train_loader=trainloader, valid_loader=valid_loader, run_id=None)
