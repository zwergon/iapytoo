
import base64
import json
from io import BytesIO
from typing import Optional

import numpy as np
from pydantic import BaseModel, field_validator


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


class MlConditionInput(BaseModel):
    """
    Charge utile d'un tableau (x, ou une condition c) : on_disk/data + (de)serialisation.

    Base de MlInput (heritage a sens unique, PAS une auto-reference) : MLflow
    infere le schema du pyfunc depuis les type hints de predict() en parcourant
    recursivement les modeles pydantic imbriques, sans detection de cycle -- un
    MlInput.condition type Optional["MlInput"] (auto-reference directe)
    declenche une RecursionError qui n'est pas rattrapee partout cote MLflow
    (elle casse purement et simplement save_mlflow_model() dans
    mlflow.models.signature._infer_signature_from_type_hints, contrairement au
    chemin de _get_func_info_if_type_hint_supported qui, lui, l'attrape
    proprement en warning -- verifie empiriquement en relancant un vrai
    entrainement). MlInput(MlConditionInput) evite le cycle : en parcourant
    MlInput.condition (type MlConditionInput), MLflow ne retrouve QUE
    on_disk/data (des scalaires), jamais MlInput lui-meme -- ce serait
    different si l'heritage etait inverse (MlConditionInput(MlInput)) ou si ce
    champ existait ici aussi. Cette classe doit donc rester le point terminal
    de la hierarchie : ne jamais lui ajouter un champ qui reference MlInput ou
    MlConditionInput.
    """
    on_disk: bool = False
    data: bytes = None

    @field_validator("data", mode="before")
    @classmethod
    def ensure_bytes(cls, v):
        if v is None:
            return v
        if isinstance(v, bytes):
            return v
        if isinstance(v, str):
            return base64.b64decode(v)
        raise TypeError("data must be bytes or base64 string")

    @property
    def path(self):
        if self.on_disk:
            return self.data.decode('utf-8')

        return ""

    @staticmethod
    def from_array(array: np.ndarray) -> "MlConditionInput":
        return MlConditionInput(on_disk=False, data=_npy_bytes(array))

    def to_array(self, context):
        if not self.on_disk:
            buffer = BytesIO(self.data)
            buffer.seek(0)
            return np.load(buffer, allow_pickle=False)
        return np.load(self.path)

    def to_bytes(self):
        return json.dumps({
            "on_disk": self.on_disk,
            "data": base64.b64encode(self.data).decode() if self.data is not None else None,
        }).encode()


class MlInput(MlConditionInput):
    """
    MlConditionInput (on_disk/data d'un tableau x) + une condition (c) optionnelle
    associee, portee a cote du tableau principal plutot que via le "params" de
    predict() : params s'applique a tout un batch/une requete (pas d'alignement
    avec les lignes de model_input dans ParamSchema/ParamSpec), alors qu'ici
    chaque echantillon peut avoir sa propre condition. condition=None pour un
    modele non conditionnel (comportement inchange). Voir la docstring de
    MlConditionInput pour pourquoi ce champ est type MlConditionInput et pas
    MlInput (auto-reference -> RecursionError cote MLflow).
    """
    condition: Optional[MlConditionInput] = None

    @staticmethod
    def from_path(path: str) -> "MlInput":
        return MlInput(on_disk=True, data=path.encode("utf-8"))

    @staticmethod
    def input_example() -> "MlInput":
        return MlInput.from_path("input_example")

    @staticmethod
    def from_array(array: np.ndarray, condition: np.ndarray = None) -> "MlInput":
        result = MlInput(on_disk=False, data=_npy_bytes(array))
        if condition is not None:
            result.condition = MlConditionInput.from_array(condition)
        return result

    def to_array(self, context):
        # Seul MlInput a besoin de resoudre le placeholder on-disk de
        # input_example() via context.artifacts -- une condition (toujours
        # embarquee via from_array, jamais enregistree comme artefact
        # input_example) n'en a jamais besoin, d'ou l'override plutot qu'un
        # comportement partage dans la base.
        if not self.on_disk:
            return super().to_array(context)
        if self.path == MlInput.input_example().path:
            path = context.artifacts[self.path]
        else:
            path = self.path
        return np.load(path)

    def to_condition_array(self, context):
        if self.condition is None:
            return None
        return self.condition.to_array(context)

    def to_bytes(self):
        return json.dumps({
            "on_disk": self.on_disk,
            "data": base64.b64encode(self.data).decode() if self.data is not None else None,
            "condition": json.loads(self.condition.to_bytes()) if self.condition is not None else None
        }).encode()
