import unittest

import torch.nn as nn
from iapytoo.train.factories import Factory
from iapytoo.train.model import Model
from iapytoo.mlflow.model import MlflowModelProvider
from iapytoo.utils.config import Config, ConfigFactory


class ModelTest(Model):
    def __init__(self, config):
        super(ModelTest, self).__init__(config)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        output = self.fc(x)
        return output


class DummyProvider(MlflowModelProvider):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._model = ModelTest(config)

    def code_definition(self) -> dict:
        return {}


class TestModel(unittest.TestCase):

    def setUp(self):

        self.config_data = {
            "project": "iapytoo",
            "run": "test_run",
            "sensors": "sensor_1",
            "model": {
                "provider": "DummyProvider"
            },
            "dataset": {
                "path": "dummy_path",
                "batch_size": 32
            },
            "training": {
                "learning_rate": 0.001,
                "loss": "mse",
                "optimizer": "adam",
                "scheduler": "step"
            }
        }

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_model(self):
        factory = Factory()
        factory.register_provider(DummyProvider)

        config = ConfigFactory.from_args(self.config_data)
        provider = factory.create_provider(
            config.model.provider, config=config)

        print(provider._model)


if __name__ == "__main__":
    unittest.main()
