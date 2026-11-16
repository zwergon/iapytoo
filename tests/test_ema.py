import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from iapytoo.train.ema import EMA, generation_state_dict
from iapytoo.train.loss import Loss
from iapytoo.train.training import LossType, Training


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))


def _perturb(model, scale=1.0):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn_like(p))


def _training(model, ema=None):
    """A Training instance without Training.__init__ (no MLflow, no provider)."""
    training = object.__new__(Training)
    loss = Loss(LossType)
    loss.reset()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    training.__dict__.update(_models=[model], _schedulers=[], loss=loss, _ema=ema,
                             _optimizers=[SimpleNamespace(torch_optimizer=optimizer)])
    return training


class TestEMA(unittest.TestCase):

    def test_update(self):
        model = _model()
        ema = EMA(model, decay=0.9)
        before = [p.clone() for p in ema.model.parameters()]
        _perturb(model)
        ema.update(model)
        for e, b, p in zip(ema.model.parameters(), before, model.parameters()):
            torch.testing.assert_close(e, 0.9 * b + 0.1 * p)
        self.assertEqual(ema.n_updates, 1)

    def test_buffers_are_copied(self):
        model = _model()
        ema = EMA(model, decay=0.999)
        model.train()
        model(torch.randn(8, 3))  # updates the BatchNorm running statistics
        ema.update(model)
        for e, b in zip(ema.model.buffers(), model.buffers()):
            self.assertTrue(torch.equal(e, b))

    def test_ema_model_is_frozen(self):
        ema = EMA(_model(), decay=0.9)
        self.assertFalse(any(p.requires_grad for p in ema.model.parameters()))
        self.assertFalse(ema.model.training)

    def test_warmup(self):
        ema = EMA(_model(), decay=0.999, warmup=True)
        self.assertAlmostEqual(ema.current_decay(), 0.1)
        ema.n_updates = 10
        self.assertAlmostEqual(ema.current_decay(), 11 / 20)
        ema.n_updates = 100000
        self.assertEqual(ema.current_decay(), 0.999)
        self.assertEqual(EMA(_model(), decay=0.999).current_decay(), 0.999)

    def test_invalid_decay(self):
        with self.assertRaises(ValueError):
            EMA(_model(), decay=1.0)

    def test_attach_updates_after_each_step_of_its_optimizer_only(self):
        model = _model()
        ema = EMA(model, decay=0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        other = torch.optim.SGD(model.parameters(), lr=0.1)
        ema.attach(optimizer, model)
        model(torch.randn(8, 3)).sum().backward()
        optimizer.step()
        optimizer.step()
        other.step()
        self.assertEqual(ema.n_updates, 2)

    def test_swapped_restores_the_weights_exactly(self):
        model = _model()
        ema = EMA(model, decay=0.9)
        _perturb(model)
        raw = {k: v.clone() for k, v in model.state_dict().items()}
        with ema.swapped(model) as swapped:
            self.assertIs(swapped, model)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, ema.model.state_dict()[k]))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, raw[k]))

    def test_swapped_restores_after_an_exception(self):
        model = _model()
        ema = EMA(model, decay=0.9)
        _perturb(model)
        raw = {k: v.clone() for k, v in model.state_dict().items()}
        with self.assertRaises(RuntimeError):
            with ema.swapped(model):
                raise RuntimeError
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, raw[k]))


class TestTrainingCheckpoint(unittest.TestCase):

    def _trained(self):
        model = _model()
        ema = EMA(model, decay=0.9)
        _perturb(model)
        ema.update(model)
        _perturb(model)
        return model, ema

    def test_no_ema_by_default(self):
        state = _training(_model()).state_dict()
        self.assertNotIn("ema_0", state)

    def test_round_trip(self):
        model, ema = self._trained()
        state = _training(model, ema).state_dict()
        self.assertIn("ema_0", state)

        other = _model()
        other_training = _training(other, EMA(other, decay=0.9))
        other_training.load_state_dict(state)
        for k, v in state["ema_0"]["model"].items():
            self.assertTrue(torch.equal(other_training._ema.model.state_dict()[k], v))
        for k, v in state["model_0"].items():
            self.assertTrue(torch.equal(other.state_dict()[k], v))
        self.assertEqual(other_training._ema.n_updates, 1)

    def test_resume_without_ema_in_checkpoint(self):
        model, _ = self._trained()
        state = _training(model).state_dict()

        other = _model()
        other_training = _training(other, EMA(other, decay=0.9))
        with self.assertLogs(level="WARNING"):
            other_training.load_state_dict(state)
        # The average restarts from the weights of the checkpoint.
        for k, v in state["model_0"].items():
            self.assertTrue(torch.equal(other_training._ema.model.state_dict()[k], v))
        self.assertEqual(other_training._ema.n_updates, 0)

    def test_load_to_predict_uses_the_ema_weights(self):
        model, ema = self._trained()
        state = _training(model, ema).state_dict()

        other = _model()
        _training(other).load_state_dict(state, only_model=True)
        for k, v in state["ema_0"]["model"].items():
            self.assertTrue(torch.equal(other.state_dict()[k], v))

    def test_generation_state_dict(self):
        model, ema = self._trained()
        with_ema = _training(model, ema).state_dict()
        without = _training(model).state_dict()
        self.assertIs(generation_state_dict(with_ema), with_ema["ema_0"]["model"])
        self.assertIs(generation_state_dict(without), without["model_0"])

    def test_ema_weights_context(self):
        model, ema = self._trained()
        training = _training(model, ema)
        raw = {k: v.clone() for k, v in model.state_dict().items()}
        with training.ema_weights():
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, ema.model.state_dict()[k]))
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, raw[k]))
        # Without EMA, the context leaves the weights alone.
        with _training(model).ema_weights():
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, raw[k]))


if __name__ == "__main__":
    unittest.main()
