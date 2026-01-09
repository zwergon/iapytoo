
import os
import numpy as np
from pathlib import Path

from iapytoo.predictions.plotters import Fake1DPlotter, DSPPlotter
from iapytoo.train.factories import Factory

from torch.utils.data import DataLoader
from iapytoo.utils.config import Config, ConfigFactory
from iapytoo.utils.model_config import GanConfig
from iapytoo.train.mlflow_model import MlfowModelProvider
from iapytoo.train.wgan import WGAN

from examples.wgan.dataset import SinDataset, LatentDataset
from examples.wgan.critic import CNN1DDiscriminator, GruDiscriminator, DFTCritic
from examples.wgan.generator import CNN1DGenerator, GruGenerator


class WindProvider(MlfowModelProvider):

    def __init__(self, config: Config):
        model_config: GanConfig = config.model
        self._input_example = np.random.rand(model_config.noise_dim)

    # override
    def code_definition(self) -> dict:
        return {
            "path": str(Path(__file__).parent / "examples"),
            "model": {
                "module": "examples.wgan.generator",
                "class": "GruGenerator"
            }
        }


class WindGan(WGAN):
    def __init__(self, config: Config):
        super().__init__(config)

    def create_mlflow_provider(self, config: Config) -> MlfowModelProvider:
        return WindProvider(config)


if __name__ == "__main__":
    config = ConfigFactory.from_yaml(os.path.join(
        os.path.dirname(__file__), "config_wgan.yml"))

    # load training data
    dataset_path = Path(__file__).parent / config.dataset.path
    trainset = SinDataset(dataset_path)

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
