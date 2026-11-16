import math
import unittest

import pydantic
import torch
import torch.nn as nn

from iapytoo.train.factories import Factory
from iapytoo.train.flow_matching import FlowMatchingTraining
from iapytoo.train.loss import Loss
from iapytoo.train.model import FlowMatchingModel, ModelError, integrate
from iapytoo.train.training import LossType
from iapytoo.utils.config import ConfigFactory
from iapytoo.utils.model_config import FlowMatchingConfig


class _Velocity(nn.Module):
    """Small velocity network, conditional when config.model.cond_dim is set."""

    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(2, 2, 3, padding=1)
        self.t_proj = nn.Linear(1, 2)
        cond_dim = config.model.cond_dim
        self.c_proj = nn.Linear(cond_dim, 2) if cond_dim else None
        self.calls_with_c = 0

    def forward(self, x, t, c=None):
        h = self.conv(x) + self.t_proj(t[:, None])[:, :, None]
        if c is not None:
            self.calls_with_c += 1
            h = h + self.c_proj(c)[:, :, None]
        return h


Factory().register_backbone("test_fm_velocity", _Velocity)


def _config(**model):
    model = {"type": "flow_matching", "provider": "none", "backbone": "test_fm_velocity",
             "signal_length": 16, **model}
    return ConfigFactory.from_args({
        "project": "iapytoo", "run": "test_fm",
        "dataset": {"path": "unused", "batch_size": 4},
        "model": model,
    })


def _training(cls, model):
    """A training instance without Training.__init__ (no MLflow, no optimizer)."""
    training = object.__new__(cls)
    loss = Loss(LossType)
    loss.reset()
    training.__dict__.update(_models=[model], device=model.device, _schedulers=[],
                             _metrics={}, loss=loss)
    return training


class _ConditionalTraining(FlowMatchingTraining):
    def _split_batch(self, batch):
        x1, c = batch
        return x1.to(self.device), c.to(self.device)


class TestFlowMatchingConfig(unittest.TestCase):

    def test_registered_by_default(self):
        config = _config()
        self.assertIsInstance(config.model, FlowMatchingConfig)
        self.assertEqual(config.model.sampler, "euler")
        self.assertEqual(config.model.n_steps, 50)

    def test_unknown_sampler_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            _config(sampler="rk4")


class TestIntegrate(unittest.TestCase):
    """dx/dt = x from x(0) = 1: x(1) = e."""

    @staticmethod
    def _error(sampler, n_steps):
        x0 = torch.ones(1, 1, 1, dtype=torch.float64)
        x1 = integrate(lambda x, t: x, x0, n_steps, sampler)
        return abs(x1.item() - math.e)

    def _order(self, sampler):
        return math.log2(self._error(sampler, 50) / self._error(sampler, 100))

    def test_euler_is_order_1(self):
        self.assertAlmostEqual(self._order("euler"), 1.0, delta=0.05)

    def test_heun_is_order_2(self):
        self.assertAlmostEqual(self._order("heun"), 2.0, delta=0.05)

    def test_unknown_sampler(self):
        with self.assertRaises(ValueError):
            integrate(lambda x, t: x, torch.ones(1, 1, 1), 10, "rk4")


class TestFlowMatchingModel(unittest.TestCase):

    def test_q_sample_lies_on_the_path(self):
        torch.manual_seed(0)
        model = FlowMatchingModel(_config())
        x1 = torch.randn(8, 2, 16)
        x_t, t, v_target = model.q_sample(x1)
        self.assertEqual(t.shape, (8,))
        # x_t + (1-t)*v_target lands on x1, x_t - t*v_target on x0.
        torch.testing.assert_close(x_t + (1 - t[:, None, None]) * v_target, x1)

    def test_state_dict_round_trip(self):
        torch.manual_seed(0)
        config = _config(sampler="heun", n_steps=5)
        model = FlowMatchingModel(config)
        other = FlowMatchingModel(config)
        other.load_state_dict(model.state_dict())
        x = torch.randn(3, 2, 16)
        self.assertTrue(torch.equal(model.evaluate_one(x), other.evaluate_one(x)))

    def test_evaluate_one_passes_the_condition(self):
        model = FlowMatchingModel(_config(cond_dim=3, n_steps=4, sampler="heun"))
        model.evaluate_one(torch.randn(2, 2, 16), c=torch.randn(2, 3))
        self.assertEqual(model.backbone.calls_with_c, 8)

    def test_evaluate_loader_stops_at_n_eval(self):
        model = FlowMatchingModel(_config(n_steps=2, n_eval=5))
        loader = [torch.randn(4, 2, 16) for _ in range(3)]
        generated = [g for g, _ in model.evaluate_loader(loader)]
        self.assertEqual(sum(g.shape[0] for g in generated), 5)

    def test_evaluate_loader_refuses_conditional(self):
        model = FlowMatchingModel(_config(cond_dim=3))
        with self.assertRaises(ModelError):
            next(model.evaluate_loader([torch.randn(4, 2, 16)]))


class TestFlowMatchingTraining(unittest.TestCase):

    def _check_validate_restores_rng(self, device):
        torch.manual_seed(0)
        model = FlowMatchingModel(_config()).to(device)
        training = _training(FlowMatchingTraining, model)
        batch = torch.randn(4, 2, 16)

        torch.manual_seed(0)
        expected = torch.randn(5, device=device)
        torch.manual_seed(0)
        first = training._inner_validate(batch, batch_idx=3)
        # Validation must not consume or reseed the global generator of the device.
        self.assertTrue(torch.equal(expected, torch.randn(5, device=device)))

        torch.manual_seed(1)
        second = training._inner_validate(batch, batch_idx=3)
        # Its draws do not depend on the incoming generator state.
        self.assertEqual(first[LossType.VALID], second[LossType.VALID])

    def test_validate_rng_cpu(self):
        self._check_validate_restores_rng("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_validate_rng_cuda(self):
        self._check_validate_restores_rng("cuda")

    def test_conditional_split_batch(self):
        model = FlowMatchingModel(_config(cond_dim=3))
        training = _training(_ConditionalTraining, model)
        training._inner_validate((torch.randn(4, 2, 16), torch.randn(4, 3)), batch_idx=0)
        self.assertEqual(model.backbone.calls_with_c, 1)


if __name__ == "__main__":
    unittest.main()
