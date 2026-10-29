from torch.utils.data import Dataset

import numpy as np


class _Dataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.dataset = np.loadtxt(filename, delimiter=",").astype(np.float32)
        self.labels_size = 1
        # self.minmax_normalize()

    def __len__(self):
        return self.dataset.shape[0]

    @property
    def signal_length(self):
        # last column is one index
        return self.dataset.shape[1] - self.labels_size

    def __getitem__(self, idx):
        # add channel [1, sequence_length]
        step = self.dataset[idx: idx + 1, :-self.labels_size]
        target = self.dataset[idx, -self.labels_size:]
        return step, target


class SinDataset(_Dataset):
    def __init__(self, filename):
        super().__init__(filename=filename)
        self.labels_size = 2
