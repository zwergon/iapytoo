import mlflow
import numpy as np
from PIL import Image
import tempfile

from iapytoo.train.inference import get_model_uri
from mlflow.tracking import MlflowClient


if __name__ == "__main__":
    import argparse
    import mlflow.pyfunc as mp
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Load an MLflow model from a run_id"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="MLflow run_id (e.g. 34d327ae3df54519bef59687fb5d7622)",
    )
    args = parser.parse_args()
    logged_model = get_model_uri(args.run_id)

    client = MlflowClient()
    run = client.get_run(args.run_id)

    noise_dim = int(run.data.params['model.noise_dim'])

    # Load model as a PyFuncModel.
    loaded_model: mp.PyFuncModel = mlflow.pyfunc.load_model(logged_model)

    model_input = np.random.rand(noise_dim)
    predicted = loaded_model.predict([model_input])
    plt.figure()
    plt.plot(predicted[0, 0, :])
    plt.savefig("prediction.jpg", dpi=300, bbox_inches="tight")
    plt.close()
