import requests

from utils import get_payload_dict

payload_dict = get_payload_dict()

payload = {
    "parameters": {
        "content_type": "mlmodelinput"
    },
    "inputs": [
        {
            "name": "input-0",
            "datatype": "BYTES",
            "shape": [len(payload_dict)],
            "data": payload_dict,
            "parameters": {
                "content_type": "mlmodelinput"
            }
        }
    ]
}


endpoint = "http://localhost:8080/v2/models/wgan/infer"
response = requests.post(endpoint, json=payload)

print(response.json())
