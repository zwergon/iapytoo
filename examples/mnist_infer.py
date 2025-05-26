from torchvision import datasets
from torch.utils.data import DataLoader
from iapytoo.predictions.plotters import ConfusionPlotter
from iapytoo.utils.config import Config, ConfigFactory, DatasetConfig

from iapytoo.train.inference import MLFlowInference
from iapytoo.train.mlflow_model import MlflowTransform


class MnistInference(MLFlowInference):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ConfusionPlotter())


if __name__ == "__main__":
    import os

    # INPUT Parameters
    config = ConfigFactory.from_yaml(os.path.join(
        os.path.dirname(__file__), "config_infer.yml"))

    inference = MnistInference(config)

    transform: MlflowTransform = inference.get_transform()

    dataset_config: DatasetConfig = config.dataset

    test_dataset = datasets.MNIST(
        dataset_config.path,
        train=False,
        transform=transform.transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config.batch_size,
        num_workers=2
    )

    inference.predict(loader=test_loader)
