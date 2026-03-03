import json
import base64
from io import BytesIO
import numpy as np
import struct

from typing import List

from pydantic import BaseModel, field_validator

from mlserver.codecs import register_input_codec, register_request_codec, NumpyCodec
from mlserver.codecs.utils import SingleInputRequestCodec
from mlserver.types import (
    RequestInput,
    ResponseOutput,
    Parameters
)
# from tritonclient.grpc import InferInput


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
        return json.dumps(self.__dict__).encode()


@register_input_codec
class MlInputCodec(NumpyCodec):
    ContentType = "mlmodelinput"
    TypeHint = MlInput

    @classmethod
    def can_decode(cls, request_input: RequestInput) -> bool:
        return (
            request_input.datatype == "BYTES"
            and request_input.parameters is not None
            and request_input.parameters.content_type == cls.ContentType
        )

    @staticmethod
    def pack_bytes_tensor(inputs: list[MlInput]) -> bytes:
        buffer = bytearray()

        for item in inputs:
            json_bytes = item.to_bytes()

            # préfixe longueur (uint32 little endian)
            buffer += struct.pack("<I", len(json_bytes))
            buffer += json_bytes

        return bytes(buffer)

    # @classmethod
    # def encode_grpc_input(cls, payload: List[MlInput], **kwargs) -> InferInput:
    #     infer_input = InferInput(
    #         name="input-0",
    #         shape=[len(payload)],
    #         datatype="BYTES"
    #     )

    #     infer_input._raw_content = cls.pack_bytes_tensor(payload)
    #     infer_input._parameters = {
    #         "content_type": "mlmodelinput"
    #     }

    #     return infer_input

    @classmethod
    def encode_input(cls, name: str, payload: List[MlInput], **kwargs) -> RequestInput:
        encoded = []

        for item in payload:
            if not isinstance(item, MlInput):
                raise ValueError(f"Expected MlModelInput, got {type(item)}")

            data_field = item.data

            # bytes -> base64 string
            if isinstance(data_field, (bytes, bytearray)):
                data_field = base64.b64encode(data_field).decode()

            encoded.append(
                {
                    "on_disk": item.on_disk,
                    "data": data_field,
                }
            )

        return RequestInput(
            name=name,
            shape=[len(encoded)],
            datatype="BYTES",
            parameters=Parameters(content_type=cls.ContentType),
            data=encoded,
        )

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> List[MlInput]:
        decoded: List[MlInput] = []

        for item in request_input.data.root:
            # cas 1: bytes JSON
            if isinstance(item, (bytes, bytearray)):
                item = json.loads(item.decode("utf-8"))

            # cas 2: string JSON
            elif isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception:
                    pass  # peut déjà être dict

            # cas 3: dict direct (ce qu'on veut supporter)
            if isinstance(item, dict):
                decoded.append(MlInput(**item))
            else:
                raise ValueError(f"Unsupported item type: {type(item)}")

        return decoded

    @classmethod
    def decode_output(cls, response_output: ResponseOutput) -> np.ndarray:
        return NumpyCodec.decode_input(response_output)  # type: ignore


@register_request_codec
class MlRequestCodec(SingleInputRequestCodec):
    """
    Decodes the first input of request as a MlModelInput.
    """

    InputCodec = MlInputCodec
    ContentType = MlInputCodec.ContentType
