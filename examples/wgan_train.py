
from pathlib import Path

from iapytoo.predictions.plotters import Fake1DPlotter, DSPPlotter
from iapytoo.train.factories import Factory

import torch
from torch.utils.data import DataLoader, random_split
from iapytoo.utils.config import Config, ConfigFactory
from iapytoo.train.wgan import WGAN

from examples.wgan.dataset import SinDataset
from examples.wgan.provider import WindProvider


class WindGan(WGAN):
    def __init__(self, config: Config):
        super().__init__(config)


if __name__ == "__main__":
    from iapytoo.utils.arguments import parse_args

    factory = Factory()
    factory.register_provider(WindProvider)

    args = parse_args()

    if args.run_id is not None:
        # Reprise : config relue depuis le run MLflow, poursuivie jusqu'a
        # --epochs (total, pas en plus) a partir du dernier checkpoint.
        config = ConfigFactory.from_run_id(args.run_id, args.tracking_uri)
        config.training.epochs = args.epochs
    else:
        config = ConfigFactory.from_yaml(args.yaml)

    # load training data
    dataset_path = Path(__file__).parent / config.dataset.path
    dataset = SinDataset(dataset_path)

    # valid_loader = donnees reelles, comme pour tout Training : WGAN y estime
    # la distance de Wasserstein (w_dist_valid), et les plotters y tirent
    # autant de signaux generes (bruit via generator.get_noise()).
    n_valid = len(dataset) // 10
    trainset, validset = random_split(
        dataset, [len(dataset) - n_valid, n_valid],
        generator=torch.Generator().manual_seed(config.seed)
    )

    trainloader = DataLoader(
        trainset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        validset, batch_size=config.dataset.batch_size, shuffle=False)

    wgan = WindGan(config)
    wgan.predictions.prediction_plotter.add(Fake1DPlotter(n_plot=2))
    wgan.predictions.prediction_plotter.add(DSPPlotter())

    wgan.fit(train_loader=trainloader, valid_loader=valid_loader, run_id=args.run_id)
