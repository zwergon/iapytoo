from enum import Enum
from iapytoo.utils.iterative_mean import Mean


class Loss:
    def __init__(self, enum_cls: type[Enum]) -> None:
        """
        enum_cls: une Enum (basée sur str) qui définit les clés de loss
        ex: LossType1 ou LossType2
        """
        self.enum_cls = enum_cls
        self.losses = {}

    def flush(self):
        for loss in self.losses.values():
            loss.flush()

    def __call__(self, key: str | Enum):
        # accepte soit l'Enum, soit directement la string
        if isinstance(key, str):
            key = self.enum_cls(key)
        try:
            return self.losses[key]
        except KeyError:
            raise Exception(f"Loss with key '{key}' not found")

    def reset(self):
        # crée toutes les entrées de la Loss en fonction de l'enum fourni
        self.losses = {lt: Mean.create("ewm") for lt in self.enum_cls}

    def state_dict(self):
        return {lt.value: loss.state_dict() for lt, loss in self.losses.items()}

    def load_state_dict(self, state_dict):
        self.losses = {}
        for lt in self.enum_cls:
            if lt.value in state_dict:
                loss = Mean.create("ewm")
                loss.load_state_dict(state_dict[lt.value])
                self.losses[lt] = loss
