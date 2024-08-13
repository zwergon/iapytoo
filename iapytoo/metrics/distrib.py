import numpy as np
from iapytoo.predictions import Predictions


class MetricsDistrib:
    def __init__(self, metric):
        self.metric = metric
        self.distrib = None

    def compute(self, predictions: Predictions, columns=None):
        Y = predictions.actual
        y_hat = predictions.outputs

        self.distrib = np.zeros(shape=(Y.shape[0], Y.shape[1]))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                self.distrib[i, j] = self.metric(Y[i, j, :], y_hat[i, j, :])

        return self.distrib
