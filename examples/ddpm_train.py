import torch
import torch.nn as nn
from torch.types import Device
from torch.utils.data import DataLoader


from iapytoo.train.nn_loss import NNLoss
from iapytoo.utils.config import Config
from iapytoo.train.scheduler import Scheduler
from iapytoo.utils.model_config import DDPMConfig

from examples.ddpm.dataset.dataset import PSDDataset
from examples.ddpm.provider import PSDProvider


class PSDLoss(NNLoss):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        ddpm_config: DDPMConfig = config.model
        self.torch_loss = nn.MSELoss()
        self.freqs = torch.fft.rfftfreq(ddpm_config.signal_length)
        self.target_psd = self.freqs**(-5/3)
        self.target_psd[0] = 0

    # override
    def to(self, device: Device):
        self.target_psd = self.target_psd.to(device)

    def __call__(self, pred, target):
        X = torch.fft.rfft(pred, dim=-1)
        psd = (X.abs()**2).mean(0).squeeze()
        return self.torch_loss(psd, self.target_psd)


class PlateauScheduler(Scheduler):

    def __init__(self, optimizer, config: Config) -> None:
        super().__init__(optimizer, config)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.loss: float = 0

    def update(self, loss: float):
        self.loss += loss

    def step(self):
        assert self.lr_scheduler is not None, f"no torch scheduler associated with {self.__class__.__name__}"
        self.lr_scheduler.step(self.loss)
        self.loss = 0


if __name__ == "__main__":
    from iapytoo.train.ddpm import DDPM
    from iapytoo.train.factories import Factory
    from iapytoo.utils.config import ConfigFactory
    from iapytoo.utils.arguments import parse_args

    factory = Factory()
    factory.register_provider(PSDProvider)
    factory.register_scheduler("plateau", PlateauScheduler)
    factory.register_loss("psd", PSDLoss)

    # INPUT Parametersfrom iapytoo.utils.arguments import parse_args

    args = parse_args()
    config = ConfigFactory.from_yaml(args.yaml)

    ddpm_config: DDPMConfig = config.model

    # -----------------------------
    # Entraînement
    # -----------------------------
    dataset = PSDDataset(T=ddpm_config.signal_length)

    loader = DataLoader(
        dataset, batch_size=config.dataset.batch_size, shuffle=True)

    training = DDPM(config)
    training.fit(loader)


# np.save("samples.npy", samples)
