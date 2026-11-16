import torch
import torch.nn.functional as F

from iapytoo.train.training import Training, LossType
from iapytoo.train.model import FlowMatchingModel
from iapytoo.utils.rng import seeded_rng


class FlowMatchingTraining(Training):
    """
    Trains a FlowMatchingModel (config.model.type: flow_matching), the way DDPM
    trains a DDPMModel. Epochs, optimizer, scheduler, logging and checkpoints
    come from Training.

    Batches are bare tensors x1 of shape (B, C, N), with no (X, Y) pair.
    Conditioning goes through _split_batch(batch) -> (x1, c), the single entry
    point: this class returns (batch, None), and a conditional subclass only
    overrides it to extract c from its own batch format.

    The model must be created with a velocity network as backbone
    (config.model.backbone), called as backbone(x_t, t) or backbone(x_t, t, c).
    Gradients are clipped to norm 1.
    """

    def _split_batch(self, batch):
        """batch -> (x1, c) on self.device. Unconditional: c = None."""
        return batch.to(self.device), None

    def _loss(self, x1, c=None):
        """
        Flow matching loss, plus a one-step reconstruction of x1 from the same
        forward pass, at no extra cost: x1_hat = x_t + (1-t)*v_pred (exact when
        v_pred == v_target). The metrics compare x1_hat to x1: a cheap proxy of
        the generation quality, not a full ODE integration (evaluate_one), which
        would be too expensive at every batch.
        """
        model: FlowMatchingModel = self.model
        x_t, t, v_target = model.q_sample(x1)
        v_pred = model(x_t, t) if c is None else model(x_t, t, c)
        loss = F.mse_loss(v_pred, v_target)
        x1_hat = x_t + (1.0 - t[:, None, None]) * v_pred.detach()
        return loss, x1_hat

    # override
    def _inner_train(self, batch, batch_idx):
        x1, c = self._split_batch(batch)

        loss, x1_hat = self._loss(x1, c)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        f_loss = loss.item()

        try:
            self._metrics["Train"].update(x1_hat, x1)
        except KeyError:
            pass

        self.loss(LossType.TRAIN).update(f_loss)
        return {LossType.TRAIN: f_loss}

    # override
    def _inner_validate(self, batch, batch_idx):
        x1, c = self._split_batch(batch)

        # Seeded by batch_idx: each validation batch gets the same (t, noise)
        # at every epoch, so the validation loss is comparable across epochs.
        # seeded_rng leaves the global CPU and CUDA generators untouched.
        with seeded_rng(42 + batch_idx, x1.device):
            loss, x1_hat = self._loss(x1, c)

        f_loss = loss.item()
        if self.scheduler is not None:
            self.scheduler.update(f_loss)

        try:
            self._metrics["Valid"].update(x1_hat, x1)
        except KeyError:
            pass

        self.loss(LossType.VALID).update(f_loss)
        return {LossType.VALID: f_loss}
