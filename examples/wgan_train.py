
import os
import numpy as np
from pathlib import Path

from iapytoo.predictions.plotters import Fake1DPlotter, DSPPlotter
from iapytoo.train.factories import Factory

from torch.utils.data import DataLoader
from iapytoo.utils.config import Config, ConfigFactory
from iapytoo.train.wgan import WGAN

from examples.wgan.dataset import SinDataset, LatentDataset
from examples.wgan.provider import WindProvider


class WindGan(WGAN):
    def __init__(self, config: Config):
        super().__init__(config)


if __name__ == "__main__":
    config = ConfigFactory.from_yaml(os.path.join(
        os.path.dirname(__file__), "config_wgan.yml"))

    # load training data
    dataset_path = Path(__file__).parent / config.dataset.path
    trainset = SinDataset(dataset_path)

    factory = Factory()
    factory.register_provider(WindProvider)

    trainloader = DataLoader(
        trainset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        drop_last=True
    )

    latentset = LatentDataset(
        config.model.noise_dim,
        size=16
    )

    valid_loader = DataLoader(
        latentset, batch_size=16, shuffle=False)

    wgan = WindGan(config)
    wgan.predictions.prediction_plotter.add(Fake1DPlotter(n_plot=2))
    wgan.predictions.prediction_plotter.add(DSPPlotter())

    wgan.fit(train_loader=trainloader, valid_loader=valid_loader, run_id=None)
