import torch
import unittest
from iapytoo.dataset import DummyVisionDataset, DummyLabelDataset
import numpy as np


class TestDataset(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_dummy_dataset(self):
        dataset = DummyVisionDataset()

        # get statistics properties of targets 1D
        print(np.mean(dataset.targets), np.std(dataset.targets))

        # images (N, C, dim, dim),
        # computes mean value for each component

        np.testing.assert_array_almost_equal(
            np.mean(dataset.images, axis=(0, 2, 3)), [7.457975, 4.4336934, 1.36766]
        )

        # computes mean value for each component
        np.testing.assert_array_almost_equal(
            np.std(dataset.images, axis=(0, 2, 3)), [4.8083544, 4.5342894, 4.8105454]
        )

        print(".values", dataset.targets)

        X, Y = dataset[0]
        # print(X)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=3, shuffle=False, drop_last=True
        )

        X, Y = next(iter(loader))
        # print(X)
        print(Y)
        print(Y.shape)

    def test_label_dataset(self):
        dataset = DummyLabelDataset()
        print("labels", dataset.targets)

        X, Y = dataset[0]
        # print(X)
        print(Y)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=3, shuffle=False, drop_last=True
        )

        X, Y = next(iter(loader))
        # print(X)
        print(Y)
        print(Y.shape)


if __name__ == "__main__":
    unittest.main()
