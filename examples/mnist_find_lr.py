from torch.utils.data import DataLoader
from torchvision import datasets

from iapytoo.dataset.transform import Transform
from iapytoo.predictions.plotters import ConfusionPlotter
from iapytoo.train.factories import Factory
from iapytoo.utils.config import ConfigFactory, Config
from iapytoo.train.training import Training

from examples.mnist.provider import MnistProvider
from examples.mnist.scheduler import MnistScheduler


if __name__ == "__main__":
    from iapytoo.utils.arguments import parse_args
    from pathlib import Path
    root_dir = Path(__file__).parent

    factory = Factory()
    factory.register_provider(MnistProvider)

    args = parse_args()

    config = ConfigFactory.from_yaml(args.yaml)

    Training.seed(config)

    training = Training(config)
    train_dataset = datasets.MNIST(
        config.dataset.path,
        train=True,
        download=True,
        transform=training.transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )

    training.find_lr(
        train_loader=train_loader
    )
