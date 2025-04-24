from torchvision import datasets
import mlflow
import numpy as np
from PIL import Image


def get_model_input(idx=0):

    dataset1 = datasets.MNIST(
        "../data",
        train=True,
        download=False
    )
    # prend une image du dataset MNIST de train
    pic: Image = dataset1[idx][0]
    # cree le numpy array
    img_input = np.array(pic.getdata()).reshape(
        1, pic.size[0], pic.size[1]).astype(np.float32)
    img_input /= 255.

    # crée un batch avec une image [1, 1, 28, 28]
    return np.array([img_input])


if __name__ == "__main__":
    run_id = "a821dbc9a0974bb0815479634c0e845c"

    logged_model = f'runs:/{run_id}/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    model_input = get_model_input()
    predicted = loaded_model.predict(model_input)
    print(predicted)
