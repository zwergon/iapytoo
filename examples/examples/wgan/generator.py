import torch
import torch.nn as nn

from iapytoo.train.model import WGANModel, WeightInitiator
from iapytoo.utils.config import Config


class CNN1DInitiator(WeightInitiator):
    def __call__(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class CNN1DGenerator(WGANModel):
    def __init__(self, config: Config):
        super(CNN1DGenerator, self).__init__(config)
        self.noise_dim = config.model.noise_dim
        self.signal_length = config.model.signal_length

        kernel_size = self.signal_length // 16  # 4 couches qui multiplient par 2

        self.main = nn.Sequential(
            nn.ConvTranspose1d(self.noise_dim, 512,
                               kernel_size, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def get_noise(self, n_samples, noise_dim, device="cpu"):
        return torch.randn(n_samples, noise_dim, device=device)

    def unsqueeze_noise(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        return noise.view(noise.shape[0], self.noise_dim, 1)

    def weight_initiator(self):
        return CNN1DInitiator()

    def forward(self, noise):
        x = self.unsqueeze_noise(noise)
        return self.main(x)


# Générateur
class Generator(WGANModel):
    def __init__(self, config: Config) -> None:
        super(Generator, self).__init__(config)
        noise_dim = config.model.noise_dim
        self.signal_length = config.model.signal_length

        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.signal_length),
        )

    def forward(self, z):
        return self.model(z)


class GruGenerator(WGANModel):
    def __init__(self, config: Config):
        super(GruGenerator, self).__init__(config)
        self.noise_dim = config.model.noise_dim
        self.hidden_size = config.model.hidden_size
        self.signal_length = config.model.signal_length

        self.gru = nn.GRU(self.noise_dim, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.signal_length)

    def get_noise(self, n_samples, noise_dim, device="cpu"):
        return torch.randn(n_samples, noise_dim, device=device)

    def unsqueeze_noise(self, noise):
        """
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns a copy of that noise with width and height = 1 and channels = z_dim.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        """
        # 'z' doit avoir la forme (batch_size, sequence_length, latent_dim)
        # at that point one noise vector with noise_dim features -> [batch, 1 , noise_dim]
        return noise.view(noise.shape[0], 1, self.noise_dim)

    def forward(self, noise):
        z = self.unsqueeze_noise(noise)
        batch_size = z.size(0)
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(z.device)
        gru_out, _ = self.gru(z, h_0)
        output = self.fc(gru_out)
        return output


# Définir le générateur
class ARGenerator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, ar_size=7):
        super(ARGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.ar_size = ar_size
        self.ar_coefficient = nn.Parameter(
            torch.randn(self.ar_size) / self.ar_size
        )  # Coefficient AR pour le bruit, Attention à la divergence.
        self.output_size = output_size

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))

        # Générer le bruit AR
        noise = self.generate_ar_noise(x.size(0), x.size(1))

        return torch.add(x, noise)  # Ajouter le bruit AR au signal généré

    def generate_ar_noise(self, batch_size, sequence_length):
        # Générer un bruit gaussien
        noise = torch.randn(batch_size, sequence_length)
        ar_noise = torch.zeros_like(noise)

        # Calculer le bruit AR en utilisant le coefficient AR
        for t in range(self.ar_size, sequence_length):
            ar_noise[:, t] = noise[:, t]
            for i in range(self.ar_size):
                ar_noise[:, t] += ar_noise[:, t - i - 1] * \
                    self.ar_coefficient[i]

        return ar_noise
