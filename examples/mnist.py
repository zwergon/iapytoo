from torch.utils.data import DataLoader
from torchvision import datasets

from iapytoo.predictions.plotters import ConfusionPlotter
from iapytoo.train.factories import Factory
from iapytoo.utils.config import ConfigFactory, Config
from iapytoo.train.training import Training
from iapytoo.train.mlflow_model import MlflowTransform, IMlfowModelProvider

from examples.subclasses import MnistModel, MnistScheduler, MnistMlfowModel


class MnistTraining(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ConfusionPlotter())
        self.mlflow_model_provider: IMlfowModelProvider = MnistMlfowModel()


if __name__ == "__main__":
    from iapytoo.utils.arguments import parse_args
    from pathlib import Path
    root_dir = Path(__file__).parent

    factory = Factory()
    factory.register_model("mnist", MnistModel)
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
    mflow_transform: MlflowTransform = training.get_transform()
    train_dataset = datasets.MNIST(
        config.dataset.path,
        train=True,
        download=True,
        transform=mflow_transform.transform
    )

    test_dataset = datasets.MNIST(
        config.dataset.path,
        train=False,
        transform=mflow_transform.transform
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
