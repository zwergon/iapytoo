

from iapytoo.predictions.plotters import Fake1DPlotter, DSPPlotter
from iapytoo.train.factories import Factory

from torch.utils.data import DataLoader
from iapytoo.utils.config import Config, ConfigFactory
from iapytoo.train.mlflow_model import IMlfowModelProvider, MlflowTransform
from iapytoo.train.wgan import WGAN
import torch
import os
from mlflow.models import ModelSignature
import numpy as np
from dataset import SinDataset, LatentDataset
from critic import CNN1DDiscriminator, GruDiscriminator, DFTCritic
from generator import CNN1DGenerator, GruGenerator


class WindProvider(IMlfowModelProvider):

    def __init__(self, config: Config):
        self.config = config

    def get_input_example(self) -> np.array:
        # Example shape for input noise signal
        return np.random.rand(self.config.model.noise_dim)


class WindGan(WGAN):
    def __init__(self, config: Config):
        super().__init__(config)
        self.mlflow_model_provider = WindProvider(config)


if __name__ == "__main__":
    config = ConfigFactory.from_yaml(os.path.join(
        os.path.dirname(__file__), "config_wgan.yml"))

    # load training data
    trainset = SinDataset()

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

    factory = Factory()
    factory.register_model("g_cnn1d", CNN1DGenerator)
    factory.register_model("d_cnn1d", CNN1DDiscriminator)
    factory.register_model("g_gru", GruGenerator)
    factory.register_model("d_gru", GruDiscriminator)
    factory.register_model("d_dft", DFTCritic)

    wgan = WindGan(config)
    wgan.predictions.prediction_plotter.add(Fake1DPlotter(n_plot=2))
    wgan.predictions.prediction_plotter.add(DSPPlotter())

    wgan.fit(train_loader=trainloader, valid_loader=valid_loader, run_id=None)
