from pathlib import Path
import matplotlib.pyplot as plt
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferResult
from utils import pack_bytes_tensor, get_payload_dict

client = grpcclient.InferenceServerClient(
    url="localhost:8081"
)


payload_dict = get_payload_dict()
infer_input = InferInput(
    name="input-0",
    shape=[len(payload_dict)],
    datatype="BYTES"
)

infer_input._raw_content = pack_bytes_tensor(payload_dict)
infer_input._parameters = {
    "content_type": "mlmodelinput"
}

# --- appel ---
response: InferResult = client.infer(
    model_name="wgan",
    inputs=[infer_input],
    parameters={"content_type": "mlmodelinput"},
)

predicted = response.as_numpy(name='output-1')
print(f"shape: {predicted.shape}")

root_path = Path(__file__).parent
plt.figure()
plt.plot(predicted[0, 0, :])
plt.savefig(root_path / "infer_grpc.jpg", dpi=300, bbox_inches="tight")
plt.close()
