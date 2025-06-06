import torch
import unittest

from iapytoo.utils.config import ConfigFactory
from iapytoo.dataset import DummyVisionDataset, DummyLabelDataset
from iapytoo.metrics.collection import MetricsCollection


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.config_data = {
            "project": "iapytoo",
            "run": "test_run",
            "sensors": "sensor_1",
            "model": {
                "type": "default",
                "model": "CNN"
            },
            "dataset": {
                "path": "dummy_path",
                "batch_size": 2
            },
            "training": {
                "learning_rate": 0.001,
                "loss": "mse",
                "optimizer": "adam",
                "scheduler": "step",
                "top_accuracy": 3
            }
        }

    @staticmethod
    def compute(Y):
        return Y + 0.1

    @staticmethod
    def compute_labels(Y, n_labels):
        print(Y.shape)
        Y_hat = torch.ones(size=Y.shape + (n_labels,))
        return Y_hat

    def test_regression(self):
        config = ConfigFactory.from_args(self.config_data)

        dataset = DummyVisionDataset()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.dataset.batch_size, shuffle=True, drop_last=True
        )

        collection = MetricsCollection("test", ["r2", "rms"], config)

        for X, Y in loader:
            Y_hat = TestMetrics.compute(Y)
            collection.update(Y_hat, Y)

        collection.compute()
        print(collection.results)

    def test_classification(self):
        config = ConfigFactory.from_args(self.config_data)

        dataset = DummyLabelDataset()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.dataset.batch_size, shuffle=True, drop_last=True
        )

        collection = MetricsCollection("test", ["accuracy"], config)

        for X, Y in loader:
            Y_hat = TestMetrics.compute_labels(Y, dataset.n_labels)
            print(Y, Y_hat)
            collection.update(Y_hat, Y)

        collection.compute()
        print(collection.results)


if __name__ == "__main__":
    unittest.main()
