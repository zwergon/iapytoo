
from abc import ABC, abstractmethod
from typing import List
from PIL import Image
import numpy as np


def to_numpy(img):
    array = np.asarray(img, dtype=np.float32)

    if array.ndim == 2:              # (H, W) → grayscale
        array = array[None, :, :]       # (1, H, W)
    elif array.ndim == 3:            # (H, W, C)
        array = array.transpose(2, 0, 1)  # (C, H, W)
    else:
        raise ValueError(f"Unexpected shape {array.shape}")
    return array


class Transform(ABC):

    @abstractmethod
    def __call__(self, y):
        pass


class Compose(Transform):

    def __init__(self, transforms: List[Transform]):
        self.transforms: List[Transform] = transforms

    def __call__(self, y):
        for t in self.transforms:
            y = t(y)
        return y


class MeanNormalize(Transform):
    def __init__(self, mean, std) -> None:
        self.mean: float = mean
        self.std: float = std

    def __call__(self, y):
        # Many datasets (MNIST, ...) provide PIL image as X
        if isinstance(y, Image.Image):
            y = to_numpy(y)
        return (y - self.mean) / self.std


class MinMaxNormalize(Transform):
    def __init__(self, y_min, y_max) -> None:
        self.y_min: float = y_min
        self.y_max: float = y_max

    def __call__(self, y):
        # Many datasets (MNIST, ...) provide PIL image as X
        if isinstance(y, Image.Image):
            y = to_numpy(y)
        return (y - self.y_min) / (self.y_max - self.y_min)
