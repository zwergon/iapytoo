import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from iapytoo.train.ddpm import DDPM, DDPM_LOSS
from iapytoo.train.loss import Loss


class _FakeDDPMModel:
    """Minimal stand-in for DDPMModel: draws (t, noise) on the data device like q_sample."""

    T = 10
    normalized_time = None

    def q_sample(self, real_data):
        noise = torch.randn_like(real_data)
        self.t = torch.randint(0, self.T, (real_data.shape[0],), device=real_data.device)
        return real_data + noise, noise

    def __call__(self, xt, t):
        return torch.zeros_like(xt)

    def predict(self, xt, pred_noise):
        return xt - pred_noise


def _fake_training(device):
    loss = Loss(DDPM_LOSS)
    loss.reset()
    return SimpleNamespace(
        model=_FakeDDPMModel(),
        device=device,
        criterion=F.mse_loss,
        _metrics={},
        _lambda=0.1,
        scheduler=None,
        loss=loss,
    )


class TestDDPMValidateRng(unittest.TestCase):

    def _check_restores_rng(self, device):
        training = _fake_training(device)
        batch = torch.ones(4, 3, 8)

        torch.manual_seed(0)
        expected = torch.randn(5, device=device)

        torch.manual_seed(0)
        DDPM._inner_validate(training, batch, batch_idx=3)
        actual = torch.randn(5, device=device)

        # Validation must not consume or reseed the global generator of the device.
        self.assertTrue(torch.equal(expected, actual))

    def test_validate_restores_cpu_rng(self):
        self._check_restores_rng("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_validate_restores_cuda_rng(self):
        self._check_restores_rng("cuda")

    def test_validate_draws_are_reproducible(self):
        training = _fake_training("cpu")
        batch = torch.ones(4, 3, 8)
        torch.manual_seed(0)
        first = DDPM._inner_validate(training, batch, batch_idx=3)
        torch.manual_seed(1)
        second = DDPM._inner_validate(training, batch, batch_idx=3)
        self.assertEqual(first[DDPM_LOSS.VALIDATE], second[DDPM_LOSS.VALIDATE])


if __name__ == "__main__":
    unittest.main()
