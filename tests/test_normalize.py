import unittest
import os

import numpy as np

import matplotlib.pyplot as plt


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class TestNormalize(unittest.TestCase):
    batch_size = 8

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_normalize(self):

        A = np.arange(27) - 4
        print(np.linalg.norm(A))

        A = A.reshape((3, 3, 3))

        print(A)

        print(np.linalg.norm(A))
        print(np.linalg.norm(A, axis=0))
        # print(normalized(A, 0))
        # print(normalized(A, 1))
        # print(normalized(A, 2))

        print(np.sqrt((4**2) + 5**2 + 14**2))
        print(np.sqrt((3**2) + 6**2 + 15**2))

        # print(normalized(np.arange(3)[:, None]))
        # print(normalized(np.arange(3)))


if __name__ == "__main__":
    unittest.main()
