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
    def create(kind: str, stats, x_columns, y_columns):
        scaling = None
        if kind == "min_max":
            scaling = MinMax(stats, x_columns=x_columns, y_columns=y_columns)
        elif kind == "normal":
            scaling = Normalize(stats, x_columns=x_columns, y_columns=y_columns)
        else:
            print(f"no way to normalize with {kind} kind -> not normalizing!")

        return scaling

    def __init__(self, stats, x_columns, y_columns) -> None:
        self.data = stats
        self.wx = np.zeros(shape=(len(Normalize.fields), len(x_columns)))
        self.wy = np.zeros(shape=(len(Normalize.fields), len(y_columns)))
        for row, f in enumerate(self.fields):
            for col, d in enumerate(x_columns):
                self.wx[row, col] = self.data[d][f]
            for col, d in enumerate(y_columns):
                self.wy[row, col] = self.data[d][f]

    def value(self, name, field):
        return self.data[name][field]

    def norm_x(self, X):
        pass

    def norm_y(self, Y):
        pass

    def unnorm_y(self, Y):
        pass


class Normalize(Scaling):
    """
    Base class that uses statistics to normalize between [0, 1] or standardize
    """

    fields = ["min", "max", "mean", "std"]

    def __init__(self, stats, x_columns, y_columns) -> None:
        super().__init__(stats, x_columns, y_columns)

    def norm_x(self, X):
        for i in range(X.shape[0]):
            X[i, :] = (X[i, :] - self.wx[OpType.MEAN, i]) / self.wx[OpType.STD, i]

    def norm_y(self, Y):
        for i in range(Y.shape[0]):
            Y[i, :] = (Y[i, :] - self.wy[OpType.MEAN, i]) / self.wy[OpType.STD, i]

    def unnorm_y(self, Y):
        for i in range(Y.shape[0]):
            Y[i, :] = Y[i, :] * self.wy[OpType.STD, i] + self.wy[OpType.MEAN, i]


class MinMax(Scaling):
    def __init__(self, stats, x_columns, y_columns) -> None:
        super().__init__(stats, x_columns, y_columns)

    def norm_x(self, X):
        for i in range(X.shape[0]):
            X[i, :] = (X[i, :] - self.wx[OpType.MIN, i]) / (
                self.wx[OpType.MAX, i] - self.wx[OpType.MIN, i]
            )

    def norm_y(self, Y):
        for i in range(Y.shape[0]):
            Y[i, :] = (Y[i, :] - self.wy[OpType.MIN, i]) / (
                self.wy[OpType.MAX, i] - self.wy[OpType.MIN, i]
            )

    def unnorm_y(self, Y):
        for i in range(Y.shape[0]):
            Y[i, :] = (
                Y[i, :] * (self.wy[OpType.MAX, i] - self.wy[OpType.MIN, i])
                + self.wy[OpType.MIN, i]
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
