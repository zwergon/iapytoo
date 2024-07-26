import torch
import unittest

import numpy as np

from iapytoo.dataset.scaling import Scaling
from iapytoo.dataset import DummyVisionDataset, DummyLabelDataset
from iapytoo.metrics.collection import MetricsCollection


class TestMetrics(unittest.TestCase):
    @staticmethod
    def compute(Y):
        return Y + 0.1

    @staticmethod
    def compute_labels(Y, n_labels):
        print(Y.shape)
        Y_hat = torch.ones_like(Y)
        return Y_hat + 1

    def test_regression(self):
        config = {"batch_size": 2}

        dataset = DummyVisionDataset()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )

        collection = MetricsCollection("test", ["r2", "rms"], config, loader)

        for X, Y in loader:
            Y_hat = TestMetrics.compute(Y)
            collection.update(Y_hat, Y)

        collection.compute()
        print(collection.results)

    def test_classification(self):
        config = {"batch_size": 3}

        dataset = DummyLabelDataset()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )

        collection = MetricsCollection("test", ["accuracy"], config, loader)

        for X, Y in loader:
            Y_hat = TestMetrics.compute_labels(Y, len(dataset))
            print(Y, Y_hat)
            collection.update(Y_hat, Y)

        collection.compute()
        print(collection.results)


if __name__ == "__main__":
    unittest.main()
