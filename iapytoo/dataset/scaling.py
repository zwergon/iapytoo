import numpy as np
from enum import IntEnum


class NormalizationType(IntEnum):
    Standard = 1
    LogMinMax = 2


class OpType(IntEnum):
    MIN = 0
    MAX = 1
    MEAN = 2
    STD = 3


class Scaling:
    """
    Base class that uses statistics to normalize between [0, 1] or standardize
    """

    fields = ["min", "max", "mean", "std"]

    @staticmethod
    def create(kind: str, stats, columns):
        scaling = None
        if kind == "min_max":
            scaling = MinMax(stats, columns=columns)
        elif kind == "normal":
            scaling = Normalize(stats, columns=columns)
        else:
            print(f"no way to normalize with {kind} kind -> not normalizing!")

        return scaling

    def __init__(self, stats, columns) -> None:
        self.data = stats
        self.weights = np.zeros(shape=(len(Normalize.fields), len(columns)))
        for row, f in enumerate(self.fields):
            for col, d in enumerate(columns):
                self.weights[row, col] = self.data[d][f]

    def value(self, name, field):
        return self.data[name][field]

    def __call__(self, x):
        pass

    def inv(self, x):
        pass


class Normalize(Scaling):
    """
    Base class that uses statistics to normalize between [0, 1] or standardize
    """

    fields = ["min", "max", "mean", "std"]

    def __init__(self, stats, columns) -> None:
        super().__init__(stats, columns)

    def __call__(self, x):
        return (x - self.weights[OpType.MEAN]) / self.weights[OpType.STD]

    def inv(self, x):
        return x * self.weights[OpType.STD] + self.weights[OpType.MEAN]


class MinMax(Scaling):
    def __init__(self, stats, columns) -> None:
        super().__init__(stats, columns)

    def __call__(self, x):
        return (x - self.weights[OpType.MIN]) / (
            self.weights[OpType.MAX] - self.weights[OpType.MIN]
        )

    def inv(self, x):
        return (
            x * (self.weights[OpType.MAX] - self.weights[OpType.MIN])
            + self.weights[OpType.MIN]
        )


# TODO
# class LogNormalize(Normalize):

#     def __init__(self, stats, x_columns, y_columns) -> None:
#         super().__init__(stats)

#     def __call__(self, X, Y):
#         x_min = np.log(self.wx[0, :])
#         x_max = np.log(self.wx[1, :])
#         return (np.log(X) - x_min) / (x_max -x_min), (Y - self.wy[2, :]) / self.wy[3, :]

#     def reverse_Y(self, Y, idx=0):
#         return Y * self.wy[3, idx] + self.wy[2, idx]
