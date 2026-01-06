
from abc import ABC, abstractmethod
from typing import List


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
        self.mean = mean
        self.std = std

    def __call__(self, y):
        return (y - self.mean) / self.std


class MinMaxNormalize(Transform):
    def __init__(self, y_min, y_max) -> None:
        self.y_min = y_min
        self.y_max = y_max

    def __call__(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)
