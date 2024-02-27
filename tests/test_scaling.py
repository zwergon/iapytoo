import unittest

import numpy as np

from iapytoo.dataset.scaling import Scaling


class TestScaling(unittest.TestCase):
    batch_size = 8

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        A = np.arange(20) - 4
        B = np.arange(20, 10, -0.5)
        self.stats = {
            "A": {
                "min": np.min(A),
                "max": np.max(A),
                "mean": np.mean(A),
                "std": np.std(A),
            },
            "B": {
                "min": np.min(B),
                "max": np.max(B),
                "mean": np.mean(B),
                "std": np.std(B),
            },
        }
        self.A = A.transpose()
        self.X = np.vstack([A, B]).transpose()
        print(self.X.shape)

    def test_normalize(self):
        print(self.X, self.stats)

        scaling = Scaling.create("min_max", self.stats, ["A"])

        scaled = scaling(self.A)
        print(scaled)
        print(scaling.inv(scaled))

        scaling = Scaling.create("normal", self.stats, ["A", "B"])
        scaled = scaling(self.X)
        print(scaled)
        print(scaling.inv(scaled))

        # print(np.linalg.norm(self.A))


if __name__ == "__main__":
    unittest.main()
