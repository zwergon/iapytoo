from torchvision import datasets
import mlflow
import numpy as np
from PIL import Image
import tempfile

from iapytoo.dataset.transform import to_numpy
from iapytoo.train.inference import get_model_uri
from iapytoo.train.mlflow_model import MlModelInput


def get_model_input(idx=0):

    dataset1 = datasets.MNIST(
        "../data",
        train=True,
        download=False
    )
    # prend une image du dataset MNIST de train
    pic: Image = dataset1[idx][0]
    expected = dataset1[idx][1]

    # cree le numpy array
    array = to_numpy(pic)

    return array, expected


if __name__ == "__main__":
    import argparse
    import mlflow.pyfunc as mp

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

    # Load model as a PyFuncModel.
    loaded_model: mp.PyFuncModel = mlflow.pyfunc.load_model(logged_model)

    with tempfile.NamedTemporaryFile(suffix='.npy') as temp:
        model_input, expected = get_model_input()
        np.save(temp.name, model_input)

        m_input = MlModelInput.from_path(temp.name)
        predicted = loaded_model.predict([m_input])
        print(predicted[0], expected)

    model_input, expected = get_model_input(1000)
    m_input = MlModelInput.from_array(model_input)
    predicted = loaded_model.predict([m_input])
    print(predicted[0], expected)
