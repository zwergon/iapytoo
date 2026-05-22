
import base64
import json
from io import BytesIO

import numpy as np
from pydantic import BaseModel, field_validator


class MlInput(BaseModel):
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
    def from_path(path: str):
        return MlInput(on_disk=True, data=path.encode("utf-8"))

    @staticmethod
    def input_example():
        return MlInput.from_path("input_example")

    @staticmethod
    def from_array(array: np.ndarray):
        buffer = BytesIO()
        np.save(buffer, array)
        return MlInput(on_disk=False, data=buffer.getvalue())

    def to_array(self, context):
        if not self.on_disk:
            buffer = BytesIO(self.data)
            buffer.seek(0)
            array = np.load(buffer, allow_pickle=False)
        else:
            if self.path == MlInput.input_example().path:
                path = context.artifacts[self.path]
            else:
                path = self.path

            array = np.load(path)
        return array

    def to_bytes(self):
        return json.dumps({
            "on_disk": self.on_disk,
            "data": base64.b64encode(self.data).decode() if self.data is not None else None
        }).encode()

