import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from iapytoo.train.factories import (
    Factory,
    OptimizerError,
    SchedulerError,
    MetricError
)
from iapytoo.train.model import Model
from iapytoo.utils.config import Config, ConfigFactory
from iapytoo.train.nn_loss import MSELoss
from iapytoo.mlflow.model import MlflowModelProvider


class DummyProvider(MlflowModelProvider):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def code_definition(self) -> dict:
        return {}


class TestFactory(unittest.TestCase):
    def setUp(self):
        self.factory = Factory()
        self.mock_model = MagicMock(spec=Model)
        self.mock_config = MagicMock(spec=Config)

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

    def test_create_loss(self):
        self.mock_config = ConfigFactory.from_args(self.config_data)

        loss = self.factory.create_loss("mse", self.mock_config)
        self.assertIsInstance(loss, MSELoss)

        with self.assertRaises(Exception):
            self.factory.create_loss("unknown_loss")

    def test_create_scheduler(self):
        config_data = self.config_data
        config_data["training"]["gamma"] = 0.1
        config_data["training"]["step_size"] = 10

        self.mock_config = ConfigFactory.from_args(config_data)

        param1 = torch.tensor([1.0], requires_grad=True)
        param2 = torch.tensor([2.0], requires_grad=True)
        parameters = [param1, param2]
        mock_optimizer = MagicMock(torch.optim.Adam(parameters))

        scheduler = self.factory.create_scheduler(
            "step", mock_optimizer, self.mock_config)
        self.assertIsNotNone(scheduler.lr_scheduler)

        with self.assertRaises(SchedulerError):
            self.factory.create_scheduler(
                "unknown_scheduler", mock_optimizer, self.mock_config)

    def test_create_optimizer(self):
        config_data = self.config_data
        config_data["training"]["learning_rate"] = 0.001
        config_data["training"]["weight_decay"] = None
        config_data["training"]["betas"] = None
        self.mock_config = ConfigFactory.from_args(config_data)

        # Créez un MagicMock pour simuler un nn.Module
        mock_module = MagicMock(spec=nn.Module)
        param1 = torch.tensor([1.0], requires_grad=True)
        param2 = torch.tensor([2.0], requires_grad=True)
        mock_module.parameters.return_value = [param1, param2]

        optimizer = self.factory.create_optimizer(
            "adam", mock_module, self.mock_config)
        self.assertIsNotNone(optimizer.torch_optimizer)

        with self.assertRaises(OptimizerError):
            self.factory.create_optimizer(
                "unknown_optimizer", self.mock_model, self.mock_config)

    def test_create_provider(self):
        self.factory.register_provider(DummyProvider)
        provider = self.factory.create_provider(
            "DummyProvider", self.mock_config)
        self.assertIsInstance(provider, DummyProvider)

    def test_create_metric(self):
        self.mock_config = ConfigFactory.from_args(self.config_data)
        metric = self.factory.create_metrics(
            "test_metrics", ["r2"], self.mock_config)
        self.assertIsNotNone(metric)

        with self.assertRaises(MetricError):
            self.factory.create_metrics(
                "test_metrics",
                ["unknown_metric"],
                self.mock_config)


if __name__ == "__main__":
    unittest.main()
