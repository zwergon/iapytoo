import unittest

import torch.nn as nn
from iapytoo.train.factories import Model, Factory


class ModelTest(Model):
    def __init__(self, loader, config):
        super(ModelTest, self).__init__(loader, config)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        output = self.fc(x)
        return output


class TestModel(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_model(self):
        factory = Factory()
        factory.register_model("test", ModelTest)

        model = Factory().create_model("test", config={}, loader=None)
        print(model)


if __name__ == "__main__":
    unittest.main()
