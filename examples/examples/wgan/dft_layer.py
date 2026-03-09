import torch
import torch.nn as nn

import numpy as np


# Définition de la couche personnalisée pour la DFT
class DFTLayer(nn.Module):
    def __init__(self, input_size, **kwarg):
        super(DFTLayer, self).__init__(**kwarg)
        self.input_size = input_size
        self.n = torch.arange(input_size).float()
        self.f = torch.arange(input_size).float().view(-1, 1)
        self.W_real = nn.Parameter(torch.cos(2 * np.pi * self.f * self.n / input_size))
        self.W_imag = nn.Parameter(torch.sin(2 * np.pi * self.f * self.n / input_size))

    def forward(self, x):
        real_part = torch.matmul(x, self.W_real)
        imag_part = torch.matmul(x, self.W_imag)
        return real_part, imag_part
