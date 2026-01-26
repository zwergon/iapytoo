import unittest

import numpy as np

from iapytoo.utils.config import ConfigFactory
from iapytoo.dataset.scaling import (
    MeanNormalize,
    MeanScalingByColumn,
    MinMaxNormalize,
    MinMaxScalingByColumn
)


class TestScaling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        A = np.arange(20) - 4
        B = np.arange(20, 10, -0.5)
        G = list(A) + list(B)
        cls.stats = [
            {
                "variable": "A",
                "stats": [
                    float(np.mean(A)),
                    float(np.std(A)),
                    float(np.min(A)),
                    float(np.max(A)),
                ],
            },
            {
                "variable": "B",
                "stats": [
                    float(np.mean(B)),
                    float(np.std(B)),
                    float(np.min(B)),
                    float(np.max(B)),
                ],
            },
        ]

        cls.global_stats = [
            {
                "variable": "global",
                "stats": [
                    float(np.mean(G)),
                    float(np.std(G)),
                    float(np.min(G)),
                    float(np.max(G)),
                ],
            },
        ]

        cls.config_data = {
            "project": "iapytoo",
            "run": "test_run",
            "sensors": "sensor_1",
            "model": {
                "provider": "DummyProvider"
            },
            "dataset": {
                "path": "dummy_path",
                "batch_size": 32,
            },
            "training": {
                "learning_rate": 0.001,
                "loss": "mse",
                "optimizer": "adam",
                "scheduler": "step"
            }
        }

        cls.X = np.vstack([A, B]).transpose()

    def test_by_columns(self):

        config_data = self.config_data.copy()
        config_data['dataset']['stats'] = self.stats
        config = ConfigFactory.from_args(config_data)

        scaling = MinMaxScalingByColumn(config)

        scaled = scaling(self.X)
        print(scaled)
        print(scaling.inv(scaled))

        scaling = MeanScalingByColumn(config)
        scaled = scaling(self.X)
        print(scaled)
        print(scaling.inv(scaled))

    def test_global_stats(self):

        config_data = self.config_data.copy()
        config_data['dataset']['stats'] = self.global_stats
        config = ConfigFactory.from_args(config_data)

        scaling = MinMaxNormalize(config)

        scaled = scaling(self.X)
        print(scaled)
        print(scaling.inv(scaled))

        scaling = MeanNormalize(config)
        scaled = scaling(self.X)
        print(scaled)
        print(scaling.inv(scaled))

        # print(np.linalg.norm(self.A))


if __name__ == "__main__":
    unittest.main()
