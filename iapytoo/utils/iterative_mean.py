import numpy as np
from threading import Lock


class Mean:
    @staticmethod
    def create(kind: str, **kwargs):
        if kind == "mean":
            return IterativeMean(**kwargs)
        elif kind == "ewm":
            return ExponentialSmoothingMean(**kwargs)
        elif kind == "dummy":
            return DummyMean(**kwargs)
        elif kind == "epoch":
            return EpochMean(**kwargs)
        else:
            raise KeyError("mean should be one of the following: [mean, ewm, dummy, epoch]")

    def __init__(self, **kwargs) -> None:
        self._value = 0.0
        self.iter = 0
        self.lock = Lock()
        self.buffer = []

    def state_dict(self):
        return {"value": self._value, "iter": self.iter}

    def flush(self):
        self.buffer = []

    def load_state_dict(self, state_dict):
        self._value = state_dict["value"]
        self.iter = state_dict["iter"]

    @property
    def value(self):
        with self.lock:
            val = self._value
        return val

    def reset(self):
        with self.lock:
            self.iter = 0
            self._value = 0.0

    def update(self):
        self.buffer.append((self.iter, self._value))

    def get_buffer(self):
        return self.buffer


class IterativeMean(Mean):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def update(self, value):
        with self.lock:
            self.iter += 1
            self._value = (value + (self.iter - 1) * self._value) / self.iter
            super().update()


class DummyMean(Mean):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def update(self, value):
        with self.lock:
            self.iter += 1
            self._value = value
            super().update()


class EpochMean(DummyMean):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epoch = 0

    def flush(self):
        self.epoch += 1  # Flush happens once per epoch
        return super().flush()

    def get_buffer(self):
        epoch_loss = [loss[1] for loss in self.buffer]
        epoch_mean = np.mean(epoch_loss)

        return [(self.epoch, epoch_mean)]


class ExponentialSmoothingMean(Mean):
    def __init__(self, alpha=0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    def update(self, value):
        with self.lock:
            self.iter += 1
            if self.iter == 1:
                self._value = value
            else:
                self._value = self.alpha * value + \
                    (1 - self.alpha) * self._value
            super().update()
