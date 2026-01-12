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
from abc import ABC, abstractmethod


from iapytoo.utils.config import Config, ModelConfig, ConfigFactory, DatasetConfigFactory
from iapytoo.train.factories import Factory, Model
from iapytoo.train.valuator import Valuator
from iapytoo.predictions.predictors import Predictor


class MlflowTransform(ABC):

    @abstractmethod
    def __init__(self, config: Config, transform=None):
        self.transform = transform

    def __call__(self, model_input, *args, **kwds) -> np.array:
        assert self.transform is not None, "use a MlTransform without a true tranform definition"
        return self.transform(model_input)


class MlfowModelProvider(ABC):
    """
    Abstract base class defining how an MLflow-deployable model is described.

    This class is responsible for providing all metadata required to package
    a trained model as an MLflow ``pyfunc`` model, including:

    - the Python code to ship with the model
    - an optional input example
    - an optional input/output transformation

    Subclasses must implement :meth:`code_definition` and define how the model
    is exposed for inference.
    """

    @abstractmethod
    def __init__(self, config: Config) -> None:
        """
        Initialize the model provider.

        Args:
            config (Config): Global experiment configuration object.
        """
        self._input_example = None
        self._transform = None

    @property
    def code_path(self) -> str:
        """
        Return the path to the Python code defining the MLflow model.

        The path is extracted from the dictionary returned by
        :meth:`code_definition`.

        Returns:
            str: Path to the code directory or file to be packaged with the
            MLflow model, or ``None`` if not defined.
        """
        code_definition = self.code_definition()
        return code_definition.get("path", None)

    @abstractmethod
    def code_definition(self) -> dict:
        """
        Describe the Python code required for MLflow model packaging.

        This method must return a dictionary describing how the inference
        code should be shipped with the MLflow model.

        Example:
            Valid code definition dictionary::

                {
                    "path": str(Path(__file__).parent),
                    "model": {
                        "module": "examples.subclasses",
                        "class": "MnistModel"
                    },
                    "transform": {
                        "module": "examples.subclasses",
                        "class": "MnistTransform"
                    }
                }


        Returns:
            dict: Code definition used during MLflow model logging.
        """
        pass

    @property
    def input_example(self) -> np.array:
        """
        Return an example input for the MLflow model.

        This example is used by MLflow to infer the model signature and to
        validate the input format during deployment.

        Returns:
            numpy.ndarray: Example input array, or ``None`` if not defined.
        """
        return self._input_example

    @property
    def transform(self) -> MlflowTransform:
        """
        Return the input/output transformation used for MLflow inference.

        The transform is applied before and/or after calling the model
        ``predict`` method when serving the model with MLflow.

        Returns:
            MlflowTransform: Transformation object, or ``None`` if not defined.
        """
        return self._transform


class _MlflowModelPrivate:

    @staticmethod
    def from_context(mlflow_model: '_MlflowModelPrivate', context: mp.PythonModelContext):
        private = _MlflowModelPrivate()

        code_definition_path = context.artifacts.get("code_definition", None)

        assert code_definition_path is not None, "code_definition artifact is required to load the model"

        with open(code_definition_path, "r") as file:
            code_definition = yaml.safe_load(file)

        if "config" in code_definition:
            module = importlib.import_module(
                code_definition["config"]["module"])
            if "dataset" in code_definition["config"]:
                key = code_definition["config"]["dataset"]
                dataset_config_cls = getattr(module, key)
                DatasetConfigFactory().register_dataset_config(
                    key, dataset_config_cls)

        config = ConfigFactory.from_yaml(context.artifacts["config"])

        module = importlib.import_module(code_definition["model"]["module"])

        assert "model" in code_definition, "a class of model is required to load the model"
        model_cls = getattr(module, code_definition["model"]["class"])

        private.model = model_cls(config=config)
        private.model.load_state_dict(torch.load(
            context.artifacts["model"], weights_only=True))

        if "transform" in code_definition:
            module = importlib.import_module(
                code_definition["transform"]["module"])
            transform_cls = getattr(
                module, code_definition["transform"]["class"])
            private.transform = transform_cls(config=config)
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

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

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
                      provider: MlfowModelProvider = None,
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
            # "code_paths": config.inference_extra_paths,
            "conda_env": None,
            "artifacts": artifacts
        }

        if supports_name():
            kwargs["name"] = f"model_step_{epoch}"
        else:
            kwargs["artifact_path"] = f"model_step_{epoch}"

        if provider is not None:
            if provider.input_example is not None:
                input_path = os.path.join(tmpdir, "input_example.npy")
                np.save(input_path, provider.input_example)

                kwargs["input_example"] = [MlflowModel.INPUT_EXAMPLE]
                artifacts["input_example"] = input_path

            code_definition = provider.code_definition()
            if code_definition:
                if provider.code_path is not None:

                    specs_path = os.path.join(tmpdir, "code_definition.yml")
                    with open(specs_path, "w") as file:
                        yaml.safe_dump(code_definition, file)
                    artifacts["code_definition"] = specs_path
                    kwargs['code_paths'] = [provider.code_path]
                else:
                    raise ValueError(
                        "code_path must be set in provider if code_definition is given"
                    )

        model_info = mp.log_model(**kwargs)
        logging.info(f"ModelURI: {model_info.model_uri}")
