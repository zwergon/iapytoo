import numpy as np
from typing import Any, Tuple
from torchvision.datasets.vision import VisionDataset


class DummyVisionDataset(VisionDataset):
    def __init__(self):
        super().__init__("./dummy")

        self.length = 10
        self.dim = 4
        self.n_channels = 3
        np.random.seed(12)
        A = np.array(range(self.dim * self.dim)).reshape((self.dim, self.dim))
        self.images = np.zeros(
            shape=(self.length, self.n_channels,
                   self.dim, self.dim), dtype=np.float32
        )
        for idx in range(self.length):
            for c in range(self.n_channels):
                self.images[idx, c, :, :] = (
                    A + np.random.normal(size=(self.dim, self.dim)) - c * 3
                )

        self.targets = np.vstack(
            [
                np.array(range(self.length)) +
                np.random.normal(size=(self.length)),
                np.array(range(self.length)) +
                np.random.normal(size=(self.length)),
            ]
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.images[index], self.targets[:, index]


class DummyLabelDataset(VisionDataset):
    def __init__(self):
        super().__init__("./dummy")

        self.length = 10
        self.dim = 4
        self.n_channels = 3
        self.n_labels = 7
        np.random.seed(12)
        A = np.array(range(self.dim * self.dim)).reshape((self.dim, self.dim))
        self.images = np.zeros(
            shape=(self.length, self.n_channels,
                   self.dim, self.dim), dtype=np.float32
        )
        for idx in range(self.length):
            for c in range(self.n_channels):
                self.images[idx, c, :, :] = (
                    A + np.random.normal(size=(self.dim, self.dim)) - c * 3
                )
        self.targets = np.random.randint(
            0, self.n_labels, dtype=np.int64, size=self.length
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return self.images[index], self.targets[index]
