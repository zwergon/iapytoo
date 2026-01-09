from torchvision import datasets
from torch.utils.data import DataLoader
from iapytoo.predictions.plotters import ConfusionPlotter
from iapytoo.utils.config import Config, ConfigFactory, DatasetConfig

from iapytoo.train.inference import MLFlowInference
from iapytoo.train.mlflow_model import MlflowTransform, MlfowModelProvider

from examples.subclasses import MnistMlfowModel


class MnistInference(MLFlowInference):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ConfusionPlotter())

    def create_mlflow_provider(self, config: Config) -> MlfowModelProvider:
        return MnistMlfowModel(config)


if __name__ == "__main__":
    import os
    from iapytoo.utils.arguments import parse_args

    args = parse_args()
    config = ConfigFactory.from_yaml(args.yaml)

    inference = MnistInference(config)

    transform: MlflowTransform = inference.transform

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
