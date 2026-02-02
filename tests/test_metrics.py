import torch
import unittest

import torch.nn.functional as F

from iapytoo.utils.config import ConfigFactory
from iapytoo.dataset import DummyVisionDataset, DummyLabelDataset
from iapytoo.metrics.metric import Metrics
from iapytoo.train.factories import Factory
from iapytoo.predictions.predictors import Predictor, MaxPredictor


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.config_data = {
            "project": "iapytoo",
            "run": "test_run",
            "sensors": "sensor_1",
            "model": {
                "type": "default",
                "provider": "provider",
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
    def compute_labels(Y, n_labels, exact=True):
        # 1. One-hot encoding
        Y_one_hot = F.one_hot(Y, num_classes=n_labels).float()

        # 2. Ajouter du bruit aléatoire (pour éviter des probabilités strictement 0 ou 1)
        noise = torch.rand_like(Y_one_hot)

        if exact:
            probabilities = Y_one_hot
        else:
            probabilities = noise

        # 3. Normaliser chaque ligne pour que la somme = 1
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        return probabilities

    def test_regression(self):
        config = ConfigFactory.from_args(self.config_data)

        dataset = DummyVisionDataset()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.dataset.batch_size, shuffle=True, drop_last=True
        )

        collection: Metrics = Factory().create_metrics(
            "test",
            ["r2", "rms"],
            config,
            predictor=Predictor())

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

        collection: Metrics = Factory().create_metrics(
            "test",
            ["accuracy", "accumul_accuracy"],
            config,
            predictor=MaxPredictor())

        for X, Y in loader:
            Y_hat = TestMetrics.compute_labels(Y, dataset.n_labels)
            collection.update(Y_hat, Y)

        print(collection.predicted, collection.target)
        collection.compute()
        print(collection.results)

        collection.reset()
        for X, Y in loader:
            Y_hat = TestMetrics.compute_labels(
                Y, dataset.n_labels, exact=False)
            collection.update(Y_hat, Y)

        print(collection.predicted, collection.target)
        collection.compute()
        print(collection.results)


if __name__ == "__main__":
    unittest.main()
