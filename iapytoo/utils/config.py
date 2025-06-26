import ast
import os
import sys
import logging
import mlflow
import yaml
from pydantic import BaseModel, BeforeValidator, Field
from pydantic_core import PydanticUndefined
import typing as t
from typing import List, Optional, Dict, Union
from typing_extensions import Annotated
from iapytoo.utils.model_config import ModelConfig, ModelConfigFactory
from iapytoo.utils.singleton import singleton

os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


def ensure_list(value, target_type):
    if isinstance(value, str):
        parsed_list = ast.literal_eval(value)
        if isinstance(parsed_list, list):
            return [target_type(v.strip()) for v in parsed_list]
        return parsed_list
    else:
        return value


class DatasetConfig(BaseModel):
    type: str = "default"
    path: str
    normalization: Optional[bool] = True
    batch_size: int
    indices: Annotated[List[int], BeforeValidator(
        lambda v: ensure_list(v, int))] = [0]
    padding: Optional[int] = 2
    image_size: Optional[int] = 224
    rotation: Optional[float] = 15
    version_number: Optional[str] = "1.0.0"
    version_type: Optional[str] = "stable"
    ratio_train_test: Optional[float] = 0.8
    num_workers: Optional[int] = 2


class TrainingConfig(BaseModel):
    type: str = "default"
    epochs: int = 10
    tqdm: Optional[bool] = True
    n_steps_by_batch: Optional[int] = 10
    loss: str
    learning_rate: float
    optimizer: Optional[str] = "adam"
    weight_decay: Optional[float] = None
    betas: Optional[Union[float, list[float]]] = None
    momentum: Optional[float] = 0.9
    scheduler: Optional[str] = "step"
    step_size: Optional[int] = 10
    gamma: Optional[float] = 0.9
    groups: Optional[int] = 1


class MetricsConfig(BaseModel):
    type: str = "default"
    names: Optional[Annotated[List[str], BeforeValidator(
        lambda v: ensure_list(v, str))]] = Field(default=None)
    top_accuracy: Optional[int] = 3


class PlottersConfig(BaseModel):
    type: str = "default"
    names: Optional[list[str]] = Field(default_factory=list)


_DataT = t.TypeVar("_DataT", bound=DatasetConfig)
_TrainingT = t.TypeVar("_TrainingT", bound=TrainingConfig)
_ModelT = t.TypeVar("_ModelT", bound=ModelConfig)
_MetricsT = t.TypeVar("_MetricsT", bound=MetricsConfig)
_PlottersT = t.TypeVar("_PlottersT", bound=PlottersConfig)


# region: Sub-factories
@singleton
class DatasetConfigFactory:
    def __init__(self) -> None:
        self.dataset_dict: dict[str, type[DatasetConfig]] = {
            "default": DatasetConfig}

    def register_dataset_config(
        self, key: str, dataset_config_cls: type[DatasetConfig]
    ) -> None:
        self.dataset_dict[key] = dataset_config_cls

    def create_dataset_config(self, kind, **kwargs) -> DatasetConfig:
        try:
            dataset_config: DatasetConfig = self.dataset_dict[kind](**kwargs)
        except KeyError:
            raise KeyError(f"Config for dataset {kind} doesn't exist")

        return dataset_config


@singleton
class TrainingConfigFactory:
    def __init__(self) -> None:
        self.training_dict: dict[str, type[TrainingConfig]] = {
            "default": TrainingConfig
        }

    def register_training_config(
        self, key: str, training_config_cls: type[TrainingConfig]
    ) -> None:
        self.training_dict[key] = training_config_cls

    def create_training_config(self, kind, **kwargs) -> TrainingConfig:
        try:
            training_config: TrainingConfig = self.training_dict[kind](
                **kwargs)
        except KeyError:
            raise KeyError(f"Config for training {kind} doesn't exist")

        return training_config


@singleton
class MetricsConfigFactory:
    def __init__(self) -> None:
        self.metrics_dict: dict[str, type[TrainingConfig]] = {
            "default": MetricsConfig}

    def register_metrics_config(
        self, key: str, metrics_config_cls: type[MetricsConfig]
    ) -> None:
        self.metrics_dict[key] = metrics_config_cls

    def create_metrics_config(self, kind: str, **kwargs) -> MetricsConfig:
        try:
            metrics_config: MetricsConfig = self.metrics_dict[kind](**kwargs)
        except KeyError:
            raise KeyError(f"Config for metrics {kind} doesn't exist")
        return metrics_config


@singleton
class PlottersConfigFactory:
    def __init__(self) -> None:
        self.plotters_dict: dict[str, type[PlottersConfig]] = {
            "default": PlottersConfig
        }

    def register_plotters_config(
        self, key: str, plotters_config_cls: type[PlottersConfig]
    ) -> None:
        self.plotters_dict[key] = plotters_config_cls

    def create_plotters_config(self, kind: str, **kwargs) -> PlottersConfig:
        try:
            plotters_config: PlottersConfig = self.plotters_dict[kind](
                **kwargs)
        except KeyError:
            raise KeyError(f"Config for plotter {kind} doesn't exist")
        return plotters_config


# endregion


# region: Config
class Config(BaseModel, t.Generic[_DataT, _TrainingT, _MetricsT, _PlottersT, _ModelT]):
    project: str
    run: str
    tracking_uri: Optional[str] = None
    sensors: Optional[str] = None
    cuda: Optional[bool] = True
    seed: Optional[int] = 42
    dataset: _DataT
    training: Optional[_TrainingT] = None
    metrics: _MetricsT = Field(default_factory=MetricsConfig)
    plotters: _PlottersT = Field(default_factory=PlottersConfig)
    model: _ModelT

    def to_flat_dict(self, exclude_unset=False) -> Dict[str, str]:
        """Export the config as a flattened key/value dictionary."""

        def flatten(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return flatten(self.model_dump(exclude_unset=exclude_unset))

    def __repr__(self) -> str:
        str = "\nConfig:\n"
        nested_dict = self.model_dump(exclude_unset=True)
        for k, v in nested_dict.items():
            str += f".{k}: {v}\n"
        str += "---------\n"
        return str

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def default_path():
        return os.path.join(os.path.dirname(__file__), "cf", "config.json")

    @staticmethod
    def default_config():
        return Config(Config.default_path())

    @staticmethod
    def test_config():
        return Config(os.path.join(os.path.dirname(__file__), "cf", "config_test.json"))

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # initialize root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if len(logger.handlers) > 0:
            logger.handlers[0].setFormatter(
                logging.Formatter("[%(levelname)s]  %(message)s")
            )
        else:
            sh = logging.StreamHandler(stream=sys.stdout)
            sh.setFormatter(logging.Formatter("[%(levelname)s]  %(message)s"))
            logger.addHandler(sh)

    @property
    def is_gan(self) -> bool:
        return self.model.model.lower() == "gan"


# endregion


# region: Config factory
class ConfigFactory:
    @staticmethod
    def from_args(kwargs: dict) -> Config:
        dataset_data: dict | DatasetConfig = kwargs["dataset"]
        if isinstance(dataset_data, dict):
            dataset_kind: str = dataset_data.get("type", "default")
            dataset_data["type"] = dataset_kind
            kwargs["dataset"] = DatasetConfigFactory().create_dataset_config(
                kind=dataset_kind, **dataset_data
            )

        training_data: dict | TrainingConfig | None = kwargs.get("training")
        if isinstance(training_data, dict):
            traininig_kind: str = training_data.get("type", "default")
            training_data["type"] = traininig_kind
            kwargs["training"] = TrainingConfigFactory().create_training_config(
                kind=traininig_kind, **training_data
            )

        metrics_data: dict | MetricsConfig | None = kwargs.get("metrics")
        if isinstance(metrics_data, dict):
            metrics_kind: str = metrics_data.get("type", "default")
            metrics_data["type"] = metrics_kind
            kwargs["metrics"] = MetricsConfigFactory().create_metrics_config(
                kind=metrics_kind, **metrics_data
            )

        plotters_data: dict | PlottersConfig | None = kwargs.get("plotters")
        if isinstance(plotters_data, dict):
            plotters_kind: str = plotters_data.get("type", "default")
            plotters_data["type"] = plotters_kind
            kwargs["plotters"] = PlottersConfigFactory().create_plotters_config(
                kind=plotters_kind, **plotters_data
            )

        model_data: dict | ModelConfig = kwargs["model"]
        if isinstance(model_data, dict):
            model_kind: str = model_data.get("type", "default")
            model_data["type"] = model_kind
            kwargs["model"] = ModelConfigFactory().create_model_config(
                model_kind, **model_data
            )

        return ConfigFactory.from_fields(**kwargs)

    @staticmethod
    def from_fields(
        *,
        dataset: _DataT,
        training: t.Optional[_TrainingT] = None,
        metrics: _MetricsT = PydanticUndefined,
        plotters: _PlottersT = PydanticUndefined,
        model: _ModelT,
        **kwargs,
    ) -> Config[_DataT, _TrainingT, _MetricsT, _PlottersT, _ModelT]:
        return Config[_DataT, _TrainingT, _MetricsT, _PlottersT, _ModelT](
            dataset=dataset,
            training=training,
            metrics=metrics,
            plotters=plotters,
            model=model,
            **kwargs,
        )

    @staticmethod
    def from_run_id(run_id, tracking_uri=None):

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        nested_dict = {}

        run = mlflow.get_run(run_id)
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)

        nested_dict["project"] = experiment.name
        nested_dict["run"] = run.info.run_name

        for raw_key, value in run.data.params.items():
            # convert raw strings to python types
            if isinstance(value, str) and value == 'None':
                continue
            keys: list[str] = raw_key.split(".")
            d: dict = nested_dict
            for raw_key in keys[:-1]:
                if raw_key not in d:
                    d[raw_key] = {}
                d = d[raw_key]
            d[keys[-1]] = value

        return ConfigFactory.from_args(nested_dict)

    @staticmethod
    def from_yaml(yaml_path):
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        return ConfigFactory.from_args(data)


# endregion
