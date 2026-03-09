import requests
from mlserver.types import InferenceResponse
from iapytoo.mlflow.codec import MlRequestCodec
from utils import get_input_list

input_list = get_input_list()

payload = MlRequestCodec.encode_request(payload=input_list)

endpoint = "http://localhost:8080/v2/models/wgan/infer"
response = requests.post(endpoint, json=payload.model_dump())

print(f"\nResponse: {response.status_code}\n")

response_payload = InferenceResponse.model_validate_json(response.text)
array = MlRequestCodec.decode_response(response_payload)
print(array.shape)
