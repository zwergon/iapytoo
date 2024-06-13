import torch
import unittest

import numpy as np

from iapytoo.dataset.scaling import Scaling
from iapytoo.dataset import DummyVisionDataset
from iapytoo.metrics.collection import MetricsCollection


class TestMetrics(unittest.TestCase):

    @staticmethod
    def compute(Y):
        return Y + .1


    def test_cat(self):
        config = {"batch_size": 2}

        dataset = DummyVisionDataset()
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True
        )
        
        collection = MetricsCollection("test", ["r2", "rms"], config, loader)

        for X, Y in loader:
            Y_hat = TestMetrics.compute(Y)
            print(Y_hat.shape)
            collection.update(Y_hat, Y)

        collection.compute()
        print(collection.results)



if __name__ == "__main__":
    unittest.main()
