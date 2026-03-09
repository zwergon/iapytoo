import torch
import numpy as np
from torch.utils.data import Dataset


class PSDDataset(Dataset):
    def __init__(self, n=2000, T=512, alpha=5/3):
        self.data = []
        freqs = np.fft.rfftfreq(T)
        psd = freqs**(-alpha)
        psd[0] = 0
        for _ in range(n):
            phase = np.exp(2j*np.pi*np.random.rand(len(psd)))
            X = np.sqrt(psd) * phase
            x = np.fft.irfft(X, n=T)
            x = x / np.std(x)
            self.data.append(x.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.tensor(self.data[i]).unsqueeze(0)

    def get_numpy(self, i):
        return self.data[i]
