import os
import sys
import importlib
import tempfile
import torch
import mlflow
import yaml
import logging
from typing import Any

import numpy as np
import mlflow.pyfunc as mp


from iapytoo.utils.config import Config, ModelConfig, ConfigFactory
from iapytoo.train.factories import Factory, Model
from iapytoo.train.valuator import Valuator
from iapytoo.predictions.predictors import Predictor


class MlflowTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, model_input, *args, **kwds) -> np.array:
        raise NotImplementedError


class IMlfowModelProvider:

    @property
    def code_path(self) -> str:
        code_definition = self.code_definition()
        return code_definition.get("path", None)

    def code_definition(self) -> dict:
        return {}

    def get_input_example(self) -> np.array:
        return None  # by default no input example

    def get_transform(self) -> MlflowTransform:
        return None  # by default no transform


class _MlflowModelPrivate:

    @staticmethod
    def from_context(mlflow_model: '_MlflowModelPrivate', context: mp.PythonModelContext):
        private = _MlflowModelPrivate()

        config = ConfigFactory.from_yaml(context.artifacts["config"])

        code_definition_path = context.artifacts.get("code_definition", None)

        assert code_definition_path is not None, "code_definition artifact is required to load the model"

        with open(code_definition_path, "r") as file:
            code_definition = yaml.safe_load(file)

        sys.path.insert(0, context.artifacts["zip"])

        module = importlib.import_module(code_definition["module"])

        assert "model_cls" in code_definition, "a class of model is required to load the model"
        model_cls = getattr(module, code_definition["model_cls"])

        private.model = model_cls(config=config)
        private.model.load_state_dict(torch.load(
            context.artifacts["model"], weights_only=True))

        if "transform_cls" in code_definition:
            transform_cls = getattr(module, code_definition["transform_cls"])
            private.transform = transform_cls()
        else:
            private.transform = None

        factory = Factory()
        private.valuator = factory.create_valuator(
            mlflow_model.metadata["valuator_key"],
            private.model,
            "cpu"
        )
        private.predictor = factory.create_predictor(
            mlflow_model.metadata['predictor_key']
        )

        if mlflow_model.metadata.get('inference_key') is not None:
            private.ml_predictor = factory.create_predictor(
                mlflow_model.metadata['inference_key'],
                mlflow_model.metadata.get('inference_args', {})
            )
        else:
            private.ml_predictor = private.predictor

        mlflow_model._private = private

    def __init__(self):
        self.model: Model = None
        self.transform = None
        self.valuator: Valuator = None
        self.predictor: Predictor = None
        self.ml_predictor: Predictor = None


class MlflowModel(mp.PythonModel):

    INPUT_EXAMPLE = "input_example"

    def __init__(self, metadata: dict = None) -> None:
        super().__init__()
        self.metadata = metadata
        self._private: _MlflowModelPrivate = None

    @property
    def transform(self):
        return self._private.transform

    @property
    def model(self):
        return self._private.model

    @property
    def valuator(self):
        return self._private.valuator

    @property
    def predictor(self):
        return self._private.predictor

    @property
    def ml_predictor(self):
        return self._private.ml_predictor

    def load_context(self, context):
        super().load_context(context)
        _MlflowModelPrivate.from_context(self, context)

    def predict(
        self,
        context: mp.PythonModelContext,
        model_input: list[str | np.ndarray],
        params: dict[str, Any] | None = None
    ):
        if not isinstance(model_input, (list, tuple)):
            raise ValueError("model_input must be a list of file paths")

        arrays = []
        for path in model_input:

            if isinstance(path, np.ndarray):
                arr = path.astype(np.float32)
            elif isinstance(path, str):
                if path == MlflowModel.INPUT_EXAMPLE:
                    assert MlflowModel.INPUT_EXAMPLE in context.artifacts, "no input example given during training"
                    arr = np.load(context.artifacts[MlflowModel.INPUT_EXAMPLE])
                else:
                    arr = np.load(path)
            else:
                raise TypeError("Unsupported input type")

            arrays.append(arr)

            batch = np.stack(arrays, axis=0).astype(np.float32)

        logging.info(f"predict called with input shape: {batch.shape}")
        if self.transform is not None:
            logging.info("predict use a transform")
            batch = self.transform(batch)

        batch_tensor = torch.from_numpy(batch)

        outputs_tensor = self.valuator.evaluate_one(batch_tensor)

        predictions = self.ml_predictor(outputs_tensor)

        return predictions


def zip_codes(code_path: dict, zip_path):
    import zipfile
    from pathlib import Path

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        base_dir = Path(code_path)
        for file in base_dir.rglob("*.py"):
            arcname = file.relative_to(base_dir)
            zipf.write(file, arcname=arcname)


def save_mlflow_model(config: Config,
                      model: Model,
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
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, "w") as file:
            yaml.safe_dump(config.model_dump(exclude_unset=True), file)

        model_path = os.path.join(tmpdir, "model.pt")
        torch.save(model.state_dict(), model_path)

        artifacts = {
            "model": model_path,
            "config": config_path
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

            code_definition = provider.code_definition()
            if code_definition:
                if provider.code_path is not None:
                    zip_path = os.path.join(tmpdir, "code.zip")
                    zip_codes(provider.code_path, zip_path)
                    artifacts["zip"] = zip_path

                    specs_path = os.path.join(tmpdir, "code_definition.yml")
                    with open(specs_path, "w") as file:
                        yaml.safe_dump(code_definition, file)
                    artifacts["code_definition"] = specs_path
                else:
                    raise ValueError(
                        "code_path must be set in provider if code_definition is given"
                    )

        model_info = mp.log_model(**kwargs)
        logging.info(f"ModelURI: {model_info.model_uri}")
