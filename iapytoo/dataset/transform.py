
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from iapytoo.utils.config import Config

from enum import IntEnum


class OpType(IntEnum):
    MEAN = 0
    STD = 1
    MIN = 2
    MAX = 3
    LAST = 4


def to_numpy(img):
    array = np.asarray(img, dtype=np.float32)

    if array.ndim == 2:              # (H, W) → grayscale
        array = array[None, :, :]       # (1, H, W)
    elif array.ndim == 3:            # (H, W, C)
        array = array.transpose(2, 0, 1)  # (C, H, W)
    else:
        raise ValueError(f"Unexpected shape {array.shape}")
    return array


class TransformError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TransformPhase(IntEnum):
    """
    Ou un Transform doit s'appliquer lors d'un appel MlflowModel.predict
    (iapytoo/mlflow/model.py) :
      - INPUT  : __call__ sur l'entree avant le forward/evaluate_one -- le cas
                 d'un modele predictif classique (entree = donnee physique
                 reelle a normaliser avant d'entrer dans le reseau). Valeur
                 par defaut, comportement historique (retrocompatible).
      - OUTPUT : inv sur la sortie generee -- le cas d'un modele generatif
                 (DDPM/FlowMatching evaluate_one) dont l'entree est deja du
                 bruit dans l'espace natif du modele (le forward-transformer
                 la corromprait), et dont seule la sortie doit etre ramenee
                 en unites physiques.
      - BOTH   : les deux (ex. un modele entree/sortie toutes deux physiques,
                 type autoencodeur/traduction).
    """
    INPUT = 0
    OUTPUT = 1
    BOTH = 2


class Transform(ABC):

    phase: TransformPhase = TransformPhase.INPUT

    @abstractmethod
    def __init__(self, config: Config):
        pass

    @abstractmethod
    def __call__(self, y) -> np.array:
        pass

    @abstractmethod
    def inv(self, y) -> np.array:
        pass


class Compose(Transform):

    def __init__(self, config: Config, names: list[str]):
        from iapytoo.train.factories import Factory
        super().__init__(config)
        self.transforms: List[Transform] = None
        try:
            factory = Factory()
            self.tranforms = [factory.create_transform(
                n, config) for n in names]
        except KeyError as er:
            print(f"Unable to create Transform : {er}")

    def __call__(self, y) -> np.array:
        for t in self.transforms:
            y = t(y)
        return y

    def inv(self, y) -> np.array:
        for t in self.transforms:
            y = t.inv(y)
        return y
