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
    expected = dataset1[idx][1]
    # cree le numpy array
    img_input = np.array(pic.getdata()).reshape(
        1, pic.size[0], pic.size[1]).astype(np.float64)
    img_input /= 255.

    # crée un batch avec une image [1, 1, 28, 28]
    return np.array([img_input]), expected


if __name__ == "__main__":
    import mlflow.pyfunc as mp
    run_id = "5bfea07ccb91434fafe6fc1def3c2011"

    logged_model = f'runs:/{run_id}/model'

    # Load model as a PyFuncModel.
    loaded_model: mp.PyFuncModel = mlflow.pyfunc.load_model(logged_model)

    model_input, expected = get_model_input()
    predicted = loaded_model.predict(model_input)
    print(predicted[0], expected)
