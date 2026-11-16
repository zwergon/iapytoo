import math

import torch
from torch.utils.data import Dataset


class SineDataset(Dataset):
    """Sines of random frequency and phase, shape (1, T): a simple target to
    check that a generative model reproduces a known distribution."""

    def __init__(self, n=2000, T=256, f_min=2.0, f_max=8.0, seed=0):
        g = torch.Generator().manual_seed(seed)
        freq = f_min + (f_max - f_min) * torch.rand(n, 1, generator=g)
        phase = 2 * math.pi * torch.rand(n, 1, generator=g)
        time = torch.linspace(0, 1, T)
        self.data = torch.sin(2 * math.pi * freq * time + phase).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
