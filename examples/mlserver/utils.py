from io import BytesIO
import numpy as np
import json
import struct
import base64
from pathlib import Path


def array_to_bytes(arr) -> bytes:
    b_array = BytesIO()
    np.save(b_array, arr)
    return b_array.getvalue()


def array_to_file(arr, file_name='array.npy') -> str:
    file_path = Path(__file__).parent / file_name
    np.save(file_path, arr, allow_pickle=False)
    return file_path.absolute().as_posix()


def pack_bytes_tensor(payload) -> bytes:
    buffer = bytearray()

    for item in payload:
        json_bytes = json.dumps(item).encode("utf-8")

        # préfixe longueur (uint32 little endian)
        buffer += struct.pack("<I", len(json_bytes))
        buffer += json_bytes

    return bytes(buffer)


def get_payload_dict():
    """This method shows how to define a payload dict without any reference to iapytoo
    MlInput

    Returns:
        dict: a list of {"on_disk": bool, "data": bytes} that can be json serializable
    """

    b_array = array_to_bytes(np.random.rand(20))
    file_path = array_to_file(np.random.rand(20))

    return [
        {
            "on_disk": True,
            "data": base64.b64encode(b"input_example").decode()
        },
        {
            "on_disk": True,
            "data": base64.b64encode(file_path.encode()).decode()
        },
        {
            "on_disk": False,
            "data": base64.b64encode(b_array).decode()
        }
    ]


def get_input_list():
    from iapytoo.mlflow.codec import MlInput

    file_path = array_to_file(np.random.rand(20))
    return [
        MlInput.input_example(),
        MlInput.from_array(np.random.rand(20)),
        MlInput.from_path(file_path)
    ]
