import base64
import json
import struct
from typing import List

import numpy as np
try:
    from mlserver.codecs import NumpyCodec, register_input_codec, register_request_codec
    from mlserver.codecs.utils import SingleInputRequestCodec
    from mlserver.types import Parameters, RequestInput, ResponseOutput
except ImportError as err:
    raise ImportError("MLServer is required for this feature. Install with: pip install mlserver." \
                      " Warning: mlserver does not work on Windows.") from err

from .mlinput import MlInput

# from tritonclient.grpc import InferInput



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
