from torch.utils.data import DataLoader

from iapytoo.train.factories import Factory
from iapytoo.train.flow_matching import FlowMatchingTraining
from iapytoo.utils.arguments import parse_args
from iapytoo.utils.config import ConfigFactory

from examples.fm.backbone import VelocityUNet1D
from examples.fm.dataset import SineDataset
from examples.fm.provider import SineFMProvider


if __name__ == "__main__":
    # Registrations come before ConfigFactory.from_yaml. The model config type
    # "flow_matching" is registered by iapytoo itself.
    factory = Factory()
    factory.register_provider(SineFMProvider)
    factory.register_backbone("velocity_unet1d", VelocityUNet1D)

    args = parse_args()
    config = ConfigFactory.from_yaml(args.yaml)

    T = config.model.signal_length
    train_loader = DataLoader(SineDataset(n=2000, T=T, seed=0),
                              batch_size=config.dataset.batch_size, shuffle=True)
    valid_loader = DataLoader(SineDataset(n=256, T=T, seed=1),
                              batch_size=config.dataset.batch_size)

    training = FlowMatchingTraining(config)
    training.fit(train_loader, valid_loader)
