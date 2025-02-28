import unittest
import tempfile
import yaml
from iapytoo.utils.config import Config, ModelConfig
from unittest.mock import patch, MagicMock

class TestConfig(unittest.TestCase):
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
                "batch_size": 32
            },
            "training": {
                "learning_rate": 0.001,
                "loss": "mse",
                "optimizer": "adam",
                "scheduler": "step"
            }
        }

        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml")
        yaml.dump(self.config_data, self.temp_file)
        self.temp_file.close()

    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)

    def test_config_loading(self):
        config = Config.create_from_yaml(self.temp_file.name)
        self.assertEqual(config.project, "iapytoo")
        self.assertEqual(config.run, "test_run")
        self.assertEqual(config.sensors, "sensor_1")
        self.assertFalse(config.dataset.normalization)
        self.assertEqual(config.training.n_steps_by_batch, 10)
        self.assertEqual(config.dataset.indices, [0])

        print(config)

    def test_config_types(self):
        config = Config.create_from_args(self.config_data)
        self.assertIsInstance(config.dataset.batch_size, int)
        self.assertIsInstance(config.training.learning_rate, float)
        self.assertIsInstance(config.cuda, bool)
        self.assertIsInstance(config.dataset.indices, list)
        self.assertIsInstance(config.model, ModelConfig)

    @patch("mlflow.get_run")
    def test_create_from_run_id(self, mock_get_run):
        config_ini = Config.create_from_args(self.config_data)
       
        flat_dict = config_ini.to_flat_dict()
        print(flat_dict)
    
        self.assertEqual(flat_dict["model.type"], "default")
        self.assertEqual(flat_dict["model.model"], "CNN")
        self.assertEqual(flat_dict["dataset.batch_size"], 32)
        self.assertEqual(flat_dict["training.learning_rate"], 0.001)
        
        # Simule la réponse de mlflow.get_run
        mock_run = MagicMock()
        mock_run.data.params = flat_dict 
        mock_get_run.return_value = mock_run

        # Appelle la méthode avec un run_id fictif
        config = Config.create_from_run_id(run_id="12345")

        # Vérifie que le dictionnaire imbriqué est correctement reconstruit
        self.assertEqual(config.model.type, "default")
        self.assertEqual(config.model.model, "CNN")
        self.assertEqual(config.dataset.batch_size, 32)
        self.assertEqual(config.training.learning_rate, 0.001)

        # Vérifie que mlflow.get_run a été appelé avec le bon run_id
        mock_get_run.assert_called_once_with("12345")

    


if __name__ == "__main__":
    unittest.main()
