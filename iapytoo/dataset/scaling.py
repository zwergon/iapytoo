import numpy as np

from PIL import Image

import logging

from iapytoo.utils.config import Config, DatasetConfig
from iapytoo.dataset.transform import (
    OpType,
    Transform,
    to_numpy
)


class MeanNormalize(Transform):
    def __init__(self, config: Config, name='global') -> None:
        super().__init__(config)
        dataset_config: DatasetConfig = config.dataset
        stats = dataset_config.statistic(name).stats
        if not stats:
            logging.error("no \"global\" statistics in config ")
            self.mean: float = 0.
            self.std: float = 1.
        else:
            self.mean: float = stats[OpType.MEAN]
            self.std: float = stats[OpType.STD]

    def __call__(self, y):
        # Many datasets (MNIST, ...) provide PIL image as X
        if isinstance(y, Image.Image):
            y = to_numpy(y)
        return (y - self.mean) / self.std

    def inv(self, y) -> np.array:
        return y*self.std + self.mean


class MinMaxNormalize(Transform):
    def __init__(self, config: Config, name='global') -> None:
        super().__init__(config)
        dataset_config: DatasetConfig = config.dataset
        stats = dataset_config.statistic(name).stats
        if not stats:
            logging.error("no \"global\" statistics in config ")
            self.y_min: float = 0.
            self.y_max: float = 1.
        else:
            self.y_min: float = stats[OpType.MIN]
            self.y_max: float = stats[OpType.MAX]

    def __call__(self, y) -> np.array:
        # Many datasets (MNIST, ...) provide PIL image as X
        if isinstance(y, Image.Image):
            y = to_numpy(y)
        return (y - self.y_min) / (self.y_max - self.y_min)

    def inv(self, y) -> np.array:
        return y * (self.y_max - self.y_min) + self.y_min


class ScalingByColumn(Transform):
    """
    Base class that uses statistics to normalize between [0, 1] or standardize
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.dataset_config: DatasetConfig = config.dataset
        columns = [s.variable for s in self.dataset_config.stats]
        self.weights = np.zeros(
            shape=(OpType.LAST, len(columns)))
        for col, d in enumerate(columns):
            stat = self.get_stat(d)
            for row in range(OpType.LAST):
                self.weights[row, col] = stat[row]

    def get_stat(self, name):
        statistic = self.dataset_config.statistic(name)
        if statistic is None:
            raise KeyError(f"statistic {name} not found")
        return statistic.stats

    def value(self, name, field: OpType):
        return self.get_stat(name)[field]


class MeanScalingByColumn(ScalingByColumn):

    def __call__(self, x):
        return (x - self.weights[OpType.MEAN]) / self.weights[OpType.STD]

    def inv(self, x):
        return x * self.weights[OpType.STD] + self.weights[OpType.MEAN]


class MinMaxScalingByColumn(ScalingByColumn):

    def __call__(self, x):
        return (x - self.weights[OpType.MIN]) / (
            self.weights[OpType.MAX] - self.weights[OpType.MIN]
        )

    def inv(self, x):
        return (
            x * (self.weights[OpType.MAX] - self.weights[OpType.MIN])
            + self.weights[OpType.MIN]
        )
