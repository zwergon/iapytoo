import torch
import torch.nn as nn

from iapytoo.utils.config import Config
from iapytoo.train.model import DDPMModel


class UNet1D(DDPMModel):
    def __init__(self, config: Config):
        super().__init__(config)
        channels = 1
        base = 64

        # --- Time embedding ---
        self.time_mlp = nn.Sequential(
            nn.Linear(1, base),
            nn.ReLU(),
            nn.Linear(base, base),
        )

        # --- Encoder ---
        self.down1 = nn.Sequential(
            nn.Conv1d(channels, base, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base, 3, padding=1),
            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(base, base*2, 4, stride=2, padding=1),  # downsample
            nn.ReLU(),
            nn.Conv1d(base*2, base*2, 3, padding=1),
            nn.ReLU()
        )

        # --- Bottleneck ---
        self.mid = nn.Sequential(
            nn.Conv1d(base*2, base*4, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base*4, base*2, 3, padding=1),
            nn.ReLU()
        )

        # --- Decoder ---
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(base*2, base, 4, stride=2,
                               padding=1),  # upsample
            nn.ReLU(),
            nn.Conv1d(base, base, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.Conv1d(base*2, base, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base, 3, padding=1),
            nn.ReLU()
        )

        # --- Output ---
        self.out = nn.Conv1d(base, channels, 1)

    def forward(self, x, t):
        """
        x : (B, C, L)
        t : (B,)
        """
        t_emb = self.time_mlp(t[:, None]).unsqueeze(-1)  # (B, base, 1)

        # --- Encoder ---
        h1 = self.down1(x) + t_emb
        h2 = self.down2(h1)

        # --- Bottleneck ---
        h3 = self.mid(h2)

        # --- Decoder ---
        u1 = self.up1(h3)
        u1 = torch.cat([u1, h1], dim=1)   # skip connection

        u2 = self.up2(u1)
        out = self.out(u2)
        return out
