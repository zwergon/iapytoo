import torch
from torchvision import datasets, transforms

from iapytoo.predictions.predictors import MaxPredictor
from iapytoo.predictions.plotters import ConfusionPlotter
from iapytoo.utils.config import Config
from iapytoo.train.inference import MLFlowInference


class MnistInference(MLFlowInference):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ConfusionPlotter())


if __name__ == "__main__":
    import os

    # INPUT Parameters
    config = Config.create_from_yaml(os.path.join(
        os.path.dirname(__file__), "config_infer.yml"))

    inference = MnistInference(config)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset2 = datasets.MNIST(
        config.dataset.path,
        train=False,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=config.dataset.batch_size,
        num_workers=2
    )

    inference.predict(loader=test_loader)
