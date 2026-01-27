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


from iapytoo.utils.config import (
    Config,
    ModelConfig
)
from iapytoo.train.model import Model
from iapytoo.predictions.predictors import Predictor
from iapytoo.dataset.transform import Transform


class ProviderError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MlflowModelProvider(ABC):
    """
    Abstract base class defining how an MLflow-deployable model is described.

    This class is responsible for providing all metadata required to package
    a trained model as an MLflow ``pyfunc`` model, including:

    - the Python code to ship with the model
    - an optional input example
    - an optional input transformation

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
        self._model: Model = None
        self._input_example: np.array = None
        self._transform: Transform = None
        self._predictor: Predictor = Predictor()  # default one
        self._ml_predictor: Predictor = None

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

                class MyProvider(MlflowModelProvider):
                    def __init__(self, config):
                        super().__init__(config)
                        self._model = MyTorchModel(config)
                        self._transform = MyTransform(config)
                        self._predictor = MyPredictor()

                    def code_definition(self):
                        return {
                            "path": str(Path(__file__).parent),
                            "provider": {
                                "module": "myproject.provider",
                                "class": "MyProvider"
                        }


        Returns:
            dict: Code definition used during MLflow model logging.
        """
        pass

    @property
    def input_example(self) -> np.array:
        """
        Return an example input for the MLflow model.

        This example is used by MLflow to
        validate the input format for deployment during training.

        Note: input_example is not used for Signature inference

        Returns:
            numpy.ndarray: Example input array, or ``None`` if not defined.
        """
        return self._input_example

    @property
    def transform(self) -> Transform:
        """
        Return the input transformation used for MLflow inference.

        The transform is applied before calling the model
        ``predict`` method when serving the model with MLflow.

        Returns:
            MlflowTransform: Transformation object, or ``None`` if not defined.
        """
        return self._transform

    @property
    def model(self) -> Model:
        return self._model

    @property
    def predictor(self) -> Predictor:
        return self._predictor

    @property
    def ml_predictor(self) -> Predictor:
        return self._ml_predictor if self._ml_predictor else self._predictor


class MlflowWGANProvider(MlflowModelProvider):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._discriminator = None

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def generator(self):
        return self._model


class MlflowModel(mp.PythonModel):

    INPUT_EXAMPLE = "input_example"

    @staticmethod
    def from_context(context: mp.PythonModelContext) -> MlflowModelProvider:
        from iapytoo.utils.config import (
            ConfigFactory,
            DatasetConfigFactory,
            TrainingConfigFactory
        )
        from iapytoo.utils.model_config import ModelConfigFactory

        code_definition_path = context.artifacts.get("code_definition", None)

        assert code_definition_path is not None, "code_definition artifact is required to load the model"

        with open(code_definition_path, "r") as file:
            code_definition = yaml.safe_load(file)

        if "config" in code_definition:
            module = importlib.import_module(
                code_definition["config"]["module"])
            config_type = code_definition["config"]["type"]
            if "dataset" in code_definition["config"]:
                key = code_definition["config"]["dataset"]
                dataset_config_cls = getattr(module, key)
                DatasetConfigFactory().register_dataset_config(
                    config_type, dataset_config_cls)
            if "training" in code_definition["config"]:
                key = code_definition["config"]["training"]
                training_config_cls = getattr(module, key)
                TrainingConfigFactory().register_training_config(
                    config_type, training_config_cls)
            if "model" in code_definition["config"]:
                key = code_definition["config"]["model"]
                model_config_cls = getattr(module, key)
                ModelConfigFactory().register_model_config(
                    config_type, model_config_cls)

        config = ConfigFactory.from_yaml(context.artifacts["config"])

        module = importlib.import_module(code_definition["provider"]["module"])

        assert "provider" in code_definition, "a class of provider is required to load the model"
        provider_cls = getattr(module, code_definition["provider"]["class"])

        provider: MlflowModelProvider = provider_cls(config=config)

        provider._model.load_state_dict(torch.load(
            context.artifacts["model"], weights_only=True))

        return provider

    def __init__(self) -> None:
        super().__init__()
        self._provider: MlflowModelProvider = None

    @property
    def transform(self):
        return self._provider.transform

    @property
    def model(self):
        return self._provider.model

    @property
    def predictor(self):
        return self._provider.predictor

    @property
    def ml_predictor(self):
        return self._provider.ml_predictor

    def load_context(self, context):
        super().load_context(context)
        self._provider = MlflowModel.from_context(context)

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

        outputs_tensor = self.model.evaluate_one(batch_tensor)

        predictions = self.ml_predictor(outputs_tensor)

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

        return predictions


def save_mlflow_model(config: Config,
                      provider: MlflowModelProvider,
                      epoch=0):

    def supports_name():
        version = tuple(map(int, mlflow.__version__.split(".")))
        return version >= (3, 0, 0)

    model = provider.model
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
            "python_model": MlflowModel(),
            "extra_pip_requirements": config.inference_pip_requirements,
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
