import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from iapytoo.train.ddpm import DDPM, DDPM_LOSS
from iapytoo.train.loss import Loss


class _FakeDDPMModel(nn.Module):
    """Minimal trainable stand-in for DDPMModel."""

    T = 10

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.t = None

    @property
    def normalized_time(self):
        return self.t.float() / self.T

    def q_sample(self, real_data):
        noise = torch.randn_like(real_data)
        self.t = torch.randint(0, self.T, (real_data.shape[0],))
        return real_data + noise, noise

    def forward(self, xt, t):
        return self.w * xt

    def predict(self, xt, pred_noise):
        return xt - pred_noise


class _RecordingScheduler:
    def __init__(self):
        self.losses = []

    def update(self, loss):
        self.losses.append(loss)


def _fake_training():
    model = _FakeDDPMModel()
    loss = Loss(DDPM_LOSS)
    loss.reset()
    return SimpleNamespace(
        model=model,
        device="cpu",
        criterion=F.mse_loss,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        _metrics={},
        _lambda=0.1,
        scheduler=_RecordingScheduler(),
        loss=loss,
    )


class TestDDPMScheduler(unittest.TestCase):
    """The scheduler is fed by the validation loss only, as in Training."""

    def test_train_does_not_update_scheduler(self):
        training = _fake_training()
        DDPM._inner_train(training, torch.ones(4, 1, 8), batch_idx=0)
        self.assertEqual(training.scheduler.losses, [])

    def test_validate_updates_scheduler(self):
        training = _fake_training()
        out = DDPM._inner_validate(training, torch.ones(4, 1, 8), batch_idx=0)
        self.assertEqual(training.scheduler.losses, [out[DDPM_LOSS.VALIDATE]])


if __name__ == "__main__":
    unittest.main()
