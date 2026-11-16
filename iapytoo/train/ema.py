import copy
from contextlib import contextmanager

import torch
import torch.nn as nn


class EMA:
    """
    Exponential moving average of the weights of a model:
    theta_ema <- decay * theta_ema + (1 - decay) * theta, after each optimizer step.

    The average lives in a frozen copy of the model (self.model), in eval mode.
    Buffers (e.g. BatchNorm statistics) are copied, not averaged. With
    warmup=True, the decay grows as min(decay, (1 + n) / (10 + n)), n being the
    number of updates done, so that the initial weights do not weigh on the
    average for the first ~1 / (1 - decay) steps.

    The training weights are left untouched: they keep driving the training,
    the validation and the scheduler. The averaged weights are only used to
    generate (see swapped).
    """

    def __init__(self, model: nn.Module, decay: float, warmup: bool = False) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.warmup = warmup
        self.n_updates = 0
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.model.eval()

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        n = self.n_updates
        return min(self.decay, (1 + n) / (10 + n))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        weight = 1.0 - self.current_decay()
        ema_params = list(self.model.parameters())
        params = [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(ema_params, params, weight)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)
        self.n_updates += 1

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        """Restarts the average from the current weights of model."""
        self.model.load_state_dict(model.state_dict())
        self.n_updates = 0

    def attach(self, optimizer: torch.optim.Optimizer, model: nn.Module):
        """Updates the average after each optimizer.step(). Returns the hook handle."""
        def hook(_optimizer, _args, _kwargs):
            self.update(model)
        return optimizer.register_step_post_hook(hook)

    @contextmanager
    def swapped(self, model: nn.Module):
        """Loads the averaged weights into model inside the block, and restores
        its own weights exactly on exit. The model object is the same, so
        anything holding it (provider, predictions) uses the averaged weights."""
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.model.state_dict())
        try:
            yield model
        finally:
            model.load_state_dict(saved)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "n_updates": self.n_updates,
            "model": self.model.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        # decay and warmup come from the config of the run being trained.
        self.n_updates = state_dict["n_updates"]
        self.model.load_state_dict(state_dict["model"])


def generation_state_dict(training_state: dict, index: int = 0) -> dict:
    """
    Weights to generate with, from the "training" entry of a checkpoint: the
    EMA weights of model `index` when the run kept an average, its raw
    weights otherwise.
    """
    ema = training_state.get(f"ema_{index}")
    if ema is not None:
        return ema["model"]
    return training_state[f"model_{index}"]
