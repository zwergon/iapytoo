import torch
import torch.nn as nn

from iapytoo.utils.config import Config


class VelocityUNet1D(nn.Module):
    """Small 1D U-Net predicting the velocity v(x_t, t) of a flow matching model.

    A plain nn.Module, registered as a backbone (config.model.backbone):
    FlowMatchingModel builds it and handles the process around it.
    """

    def __init__(self, config: Config):
        super().__init__()
        channels = config.model.n_channels
        base = config.model.base

        self.time_mlp = nn.Sequential(nn.Linear(1, base), nn.SiLU(), nn.Linear(base, base))
        self.down1 = nn.Sequential(
            nn.Conv1d(channels, base, 3, padding=1), nn.SiLU(),
            nn.Conv1d(base, base, 3, padding=1), nn.SiLU())
        self.down2 = nn.Sequential(
            nn.Conv1d(base, base * 2, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv1d(base * 2, base * 2, 3, padding=1), nn.SiLU())
        self.mid = nn.Sequential(
            nn.Conv1d(base * 2, base * 2, 3, padding=1), nn.SiLU())
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(
            nn.Conv1d(base * 2, base, 3, padding=1), nn.SiLU(),
            nn.Conv1d(base, base, 3, padding=1), nn.SiLU())
        self.out = nn.Conv1d(base, channels, 1)

    def forward(self, x, t):
        """x: (B, C, L), t: (B,) in [0, 1]."""
        t_emb = self.time_mlp(t[:, None]).unsqueeze(-1)
        h1 = self.down1(x) + t_emb
        h2 = self.mid(self.down2(h1))
        u1 = torch.cat([self.up1(h2), h1], dim=1)
        return self.out(self.up2(u1))
