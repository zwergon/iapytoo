from threading import Lock


class Mean:
    @staticmethod
    def create(kind: str, **kwargs):
        if kind == "mean":
            return IterativeMean(**kwargs)
        elif kind == "ewm":
            return ExponentialSmoothingMean(**kwargs)
        else:
            raise KeyError(f"mean should be mean, or ewm")

    def __init__(self, **kwargs) -> None:
        self._value = 0.0
        self.iter = 0
        self.lock = Lock()

    def state_dict(self):
        return {"value": self._value, "iter": self.iter}

    def load_state_dict(self, state_dict):
        self._value = state_dict["value"]
        self.iter = state_dict["iter"]

    @property
    def value(self):
        with self.lock:
            val = self._value
        return val

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
        pass


class IterativeMean(Mean):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def update(self, value):
        with self.lock:
            self.iter += 1
            self._value = (value + (self.iter - 1) * self._value) / self.iter


class ExponentialSmoothingMean(Mean):
    def __init__(self, alpha=0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    def update(self, value):
        with self.lock:
            self.iter += 1
            self._value = self.alpha * value + (1 - self.alpha) * self._value
