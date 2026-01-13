import torch
import torch.nn as nn

from iapytoo.train.model import Model
from iapytoo.utils.config import Config

from wind_gan.dft_layer import DFTLayer
from wind_gan.generator import CNN1DInitiator


class CNN1DDiscriminator(Model):
    def __init__(self, config: Config):
        super(CNN1DDiscriminator, self).__init__(config)
        self.signal_length = config.model.signal_length

        kernel_size = self.signal_length // 16  # 4 couches qui multiplient par 2
        self.main = nn.Sequential(
            # input kernel_size*16
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size kernel_size*8
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size kernel_size*4
            nn.Conv1d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size kernel_size*2
            nn.Conv1d(256, 512, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size kernel_size
            nn.Conv1d(512, 1, kernel_size=kernel_size,
                      stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def weight_initiator(self):
        return CNN1DInitiator()

    def forward(self, x, y=None):
        x = self.main(x)
        return x


class GruDiscriminator(Model):
    def __init__(self, config: Config):
        super(GruDiscriminator, self).__init__(config)
        self.hidden_size = config.model.hidden_size
        self.signal_length = config.model.signal_length
        self.gru = nn.GRU(self.signal_length,
                          self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # 'x' doit avoir la forme (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(
            x.device)  # Initial hidden state
        gru_out, _ = self.gru(x, h_0)
        # Utiliser la dernière sortie de la séquence
        output = self.fc(gru_out[:, -1, :])
        return output


# Discriminateur utilisant la DFTLayer
class DFTCritic(Model):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.signal_length = config.model.signal_length
        self.dft_layer = DFTLayer(self.signal_length)
        self.model = nn.Sequential(
            nn.Linear(self.signal_length, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def to(self, device):
        self.dft_layer = self.dft_layer.to(device)
        return super().to(device)

    def forward(self, x):
        real_part, imag_part = self.dft_layer(x)
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        magnitude = magnitude.view(magnitude.size(0), -1)  # Flatten
        return self.model(magnitude)
