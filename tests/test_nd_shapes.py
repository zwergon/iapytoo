import unittest

import torch
import torch.nn as nn

from iapytoo.train.factories import Factory
from iapytoo.train.flow_matching import FlowMatchingTraining
from iapytoo.train.loss import Loss
from iapytoo.train.model import DDPMModel, FlowMatchingModel, per_sample
from iapytoo.train.training import LossType
from iapytoo.utils.config import ConfigFactory

# Batch size equal to every spatial size: a coefficient broadcast along the
# wrong axis (e.g. t[:, None, None] on a volume) then raises no error, so the
# tests check the values sample by sample.
B = 4
SHAPES = {"1d": (B, 2, B), "2d": (B, 2, B, B), "3d": (B, 2, B, B, B)}


class _Zero(nn.Module):
    """Backbone of any dimension: predicts zeros, with one trainable weight."""

    def __init__(self, config):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x, t, c=None):
        return self.w * x


Factory().register_backbone("test_nd_zero", _Zero)


def _model(cls, type_, **model):
    config = ConfigFactory.from_args({
        "project": "iapytoo", "run": "test_nd",
        "dataset": {"path": "unused", "batch_size": B},
        "model": {"type": type_, "provider": "none", "backbone": "test_nd_zero", **model},
    })
    return cls(config)


class TestPerSample(unittest.TestCase):

    def test_shapes(self):
        t = torch.arange(B, dtype=torch.float32)
        for name, shape in SHAPES.items():
            with self.subTest(name):
                x = torch.ones(shape)
                y = per_sample(t, x) * x
                for i in range(B):
                    self.assertTrue(torch.all(y[i] == t[i]))


class TestFlowMatchingND(unittest.TestCase):

    def test_q_sample_is_per_sample(self):
        model = _model(FlowMatchingModel, "flow_matching")  # no signal_length needed
        for name, shape in SHAPES.items():
            with self.subTest(name):
                x1 = torch.randn(shape)
                torch.manual_seed(0)
                x_t, t, v = model.q_sample(x1)
                torch.manual_seed(0)
                x0 = torch.randn_like(x1)
                self.assertEqual(x_t.shape, x1.shape)
                for i in range(B):
                    torch.testing.assert_close(x_t[i], (1 - t[i]) * x0[i] + t[i] * x1[i])
                torch.testing.assert_close(v, x1 - x0)

    def test_loss_and_generation(self):
        model = _model(FlowMatchingModel, "flow_matching", n_steps=2, sampler="heun")
        training = object.__new__(FlowMatchingTraining)
        loss = Loss(LossType)
        loss.reset()
        training.__dict__.update(_models=[model], device=model.device, _schedulers=[],
                                 _metrics={}, loss=loss)
        for name, shape in SHAPES.items():
            with self.subTest(name):
                x1 = torch.randn(shape)
                torch.manual_seed(0)
                _, x1_hat = training._loss(x1)
                torch.manual_seed(0)
                x_t, t, _ = model.q_sample(x1)
                # The zero backbone predicts v = 0: x1_hat = x_t, per sample.
                torch.testing.assert_close(x1_hat, x_t)
                self.assertEqual(model.evaluate_one(torch.randn(shape)).shape, shape)


class TestDDPMND(unittest.TestCase):

    def test_q_sample_and_predict_are_per_sample(self):
        model = _model(DDPMModel, "ddpm", n_times=10)
        for name, shape in SHAPES.items():
            with self.subTest(name):
                x0 = torch.randn(shape)
                torch.manual_seed(0)
                xt, noise = model.q_sample(x0)
                a = model.alphas_cumprod[model.t]
                for i in range(B):
                    torch.testing.assert_close(xt[i], a[i].sqrt() * x0[i] + (1 - a[i]).sqrt() * noise[i])
                # predict inverts q_sample when the predicted noise is exact.
                torch.testing.assert_close(model.predict(xt, noise), x0, rtol=1e-4, atol=1e-4)
                self.assertEqual(model.evaluate_one(torch.randn(shape)).shape, shape)


if __name__ == "__main__":
    unittest.main()
