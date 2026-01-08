import mlflow
import numpy as np
from PIL import Image
import tempfile

from iapytoo.dataset.transform import to_numpy
from generator import GruGenerator
from mlflow.tracking import MlflowClient


if __name__ == "__main__":
    import mlflow.pyfunc as mp
    import matplotlib.pyplot as plt
    run_id = "6fcdac611218488892a89222e0dfcacb"

    logged_model = f'runs:/{run_id}/model_step_0'

    client = MlflowClient()
    run = client.get_run(run_id)

    noise_dim = int(run.data.params['model.noise_dim'])

    # Load model as a PyFuncModel.
    loaded_model: mp.PyFuncModel = mlflow.pyfunc.load_model(logged_model)

    model_input = np.random.rand(noise_dim)
    predicted = loaded_model.predict([model_input])
    plt.figure()
    plt.plot(predicted[0, 0, :])
    plt.savefig("prediction.jpg", dpi=300, bbox_inches="tight")
    plt.close()
