from torch.utils.data import DataLoader
from torchvision import datasets

from iapytoo.predictions.plotters import ConfusionPlotter
from iapytoo.train.factories import Factory
from iapytoo.utils.config import ConfigFactory, Config
from iapytoo.train.training import Training

from examples.mnist.provider import MnistProvider
from examples.mnist.scheduler import MnistScheduler


class MnistTraining(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ConfusionPlotter())


if __name__ == "__main__":
    from iapytoo.utils.arguments import parse_args

    factory = Factory()
    factory.register_provider(MnistProvider)
    factory.register_scheduler("mnist", MnistScheduler)

    args = parse_args()

    if args.run_id is not None:
        config = ConfigFactory.from_run_id(args.run_id, args.tracking_uri)
        config.training.epochs = args.epochs
    else:
        # INPUT Parameters
        config = ConfigFactory.from_yaml(args.yaml)

    Training.seed(config)

    training = MnistTraining(config)
    train_dataset = datasets.MNIST(
        config.dataset.path,
        train=True,
        download=True,
        transform=training.transform
    )

    test_dataset = datasets.MNIST(
        config.dataset.path,
        train=False,
        transform=training.transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )

    training.fit(
        train_loader=train_loader,
        valid_loader=test_loader,
        run_id=args.run_id
    )
