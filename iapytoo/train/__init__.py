import os
import tempfile
import torch
import mlflow
import numpy as np
import mlflow.pyfunc as mp

from iapytoo.utils.config import Config, ModelConfig
from .mlflow_model import IMlfowModelProvider, MlflowModel


def save_mlflow_model(config: Config,
                      model,
                      provider: IMlfowModelProvider = None,
                      epoch=0):

    model_config: ModelConfig = config.model
    metadata = {
        "valuator_key":  model_config.valuator,
        "predictor_key": model_config.predictor,
        "inference_key": model_config.inference_predictor,
        "inference_args": model_config.inference_predictor_args
    }

    def supports_name():
        version = tuple(map(int, mlflow.__version__.split(".")))
        return version >= (3, 0, 0)

    model.cpu()
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pt")
        torch.save({
            "model": model,
            "transform": provider.get_transform() if provider is not None else None
        }, model_path)

        artifacts = {
            "model": model_path
        }

        kwargs = {
            "python_model": MlflowModel(metadata),
            "extra_pip_requirements": config.inference_pip_requirements,
            "code_paths": config.inference_extra_paths,
            "conda_env": None,
            "artifacts": artifacts
        }

        if supports_name():
            kwargs["name"] = f"model_step_{epoch}"
        else:
            kwargs["artifact_path"] = f"model_step_{epoch}"

        if provider is not None:
            input_example = provider.get_input_example()
            if input_example is not None:
                input_path = os.path.join(tmpdir, "input_example.npy")
                np.save(input_path, input_example)

                kwargs["input_example"] = [MlflowModel.INPUT_EXAMPLE]
                artifacts["input_example"] = input_path

        mp.log_model(**kwargs)
