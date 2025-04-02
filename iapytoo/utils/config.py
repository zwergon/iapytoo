import ast
import os
import sys
import logging
import mlflow
import tempfile
import yaml
from pydantic import BaseModel, BeforeValidator, SerializeAsAny, Field
import typing as t
from typing import List, Optional, Dict
from typing_extensions import Annotated
from iapytoo.utils.model_config import ModelConfig, ModelConfigFactory

os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


def ensure_list(value):
    if isinstance(value, str):
        v_list = value[1:-1].split(",")
        return [int(v) for v in v_list]
    else:
        return value


class DatasetConfig(BaseModel):
    path: str
    normalization: Optional[bool] = True
    batch_size: int
    indices: Annotated[List[int], BeforeValidator(ensure_list)] = [0]
    padding: Optional[int] = 2
    image_size: Optional[int] = 224
    rotation: Optional[float] = 15
    version_number: Optional[str] = "1.0.0"
    version_type: Optional[str] = "stable"


class TrainingConfig(BaseModel):
    epochs: int = 10
    tqdm: Optional[bool] = True
    n_steps_by_batch: Optional[int] = 10
    ratio_train_test: Optional[float] = 0.8
    num_workers: Optional[int] = 2
    loss: str
    learning_rate: float
    optimizer: Optional[str] = "adam"
    weight_decay: Optional[float] = None
    betas:  Optional[float] = None
    momentum: Optional[float] = 0.9
    scheduler: Optional[str] = "step"
    step_size: Optional[int] = 10
    gamma: Optional[float] = 0.9
    groups: Optional[int] = 1


class MetricsConfig(BaseModel):
    names: Optional[List[str]] = Field(default=[])
    top_accuracy: Optional[int] = 3

DataT = t.TypeVar("DataT", bound = DatasetConfig)
TrainingT = t.TypeVar("TrainingT", bound = TrainingConfig)
MetricsT = t.TypeVar("MetricsT", bound = MetricsConfig)
ModelT = t.TypeVar("ModelT", bound = ModelConfig)

class MetricsConfigFactory:
    def __init__(self) -> None:
        self.metrics_dict: dict[str, type[MetricsConfig]] = {
            "default": MetricsConfig
        }
    
    def register_metrics_config(
        self, key: str, metrics_config_cls: type[MetricsConfig]
    ) -> None:
        self.metrics_dict[key] = metrics_config_cls

    def get_union_type(self):
        return t.Union[tuple(v for v in self.metrics_dict.values())]

    def create_metrics_config(self, kind, **kwargs) -> MetricsConfig:
        try:
            metrics_config: MetricsConfig = self.metrics_dict[kind](**kwargs)
        except KeyError:
            raise KeyError(f"Config for metrics {kind} doesn't exist")

        return metrics_config

class TrainingConfigFactory:
    def __init__(self) -> None:
        self.training_dict: dict[str, type[TrainingConfig]] = {
            "default": TrainingConfig
        }
    
    def register_training_config(
        self, key: str, training_config_cls: type[TrainingConfig]
    ) -> None:
        self.training_dict[key] = training_config_cls

    def get_union_type(self):
        return t.Union[tuple(v for v in self.training_dict.values())]

    def create_training_config(self, kind, **kwargs) -> TrainingConfig:
        try:
            training_config: TrainingConfig = self.training_dict[kind](**kwargs)
        except KeyError:
            raise KeyError(f"Config for training {kind} doesn't exist")

        return training_config

class DatasetConfigFactory:
    def __init__(self) -> None:
        self.dataset_dict: dict[str, type[DatasetConfig]] = {
            "default": DatasetConfig
        }
    
    def register_dataset_config(
        self, key: str, dataset_config_cls: type[DatasetConfig]
    ) -> None:
        self.dataset_dict[key] = dataset_config_cls

    def get_union_type(self):
        return t.Union[tuple(v for v in self.dataset_dict.values())]

    def create_dataset_config(self, kind, **kwargs) -> DatasetConfig:
        try:
            dataset_config: DatasetConfig = self.dataset_dict[kind](**kwargs)
        except KeyError:
            raise KeyError(f"Config for dataset {kind} doesn't exist")

        return dataset_config

class Config(BaseModel, t.Generic[DataT, TrainingT, MetricsT, ModelT]):
    project: str
    run: str
    tracking_uri: Optional[str] = None
    sensors: Optional[str] = None
    cuda: Optional[bool] = True
    seed: Optional[int] = 42
    dataset: DataT
    training: Optional[TrainingT] = None
    metrics: SerializeAsAny[MetricsT] = Field(default=MetricsConfig())
    model: ModelT

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

    @classmethod
    def create_from_args(cls, args: dict):

        model_data = args.pop("model")
        if "type" not in model_data:
            model_data['type'] = "default"
        model_type = model_data["type"]

        factory = ModelConfigFactory()
        cls.__annotations__["model"] = factory.model_dict[model_type]
        cls.model_fields['model'].annotation = factory.model_dict[model_type]
        cls.model_rebuild(force=True)

        model = factory.create_model_config(model_type, **model_data)

        return cls(**args, model=model)

    @classmethod
    def create_from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        return cls.create_from_args(data)

    @classmethod
    def create_from_run_id(cls, run_id, tracking_uri=None):
        cf = {}

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        nested_dict = {}

        run = mlflow.get_run(run_id)
        experiment_id = run.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)

        nested_dict['project'] = experiment.name
        nested_dict['run'] = run.info.run_name

        for raw_key, raw_value in run.data.params.items():
            try:
                value = eval(raw_value)
            except (SyntaxError, NameError):
                value = raw_value
            if value is None:
                continue
            keys: list[str] = raw_key.split(".")
            d: dict = nested_dict
            for raw_key in keys[:-1]:
                if raw_key not in d:
                    d[raw_key] = {}
                d = d[raw_key]
            d[keys[-1]] = value

        return cls.create_from_args(nested_dict)

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
    
class ConfigFactory:
    @staticmethod
    def from_fields[DataT, TrainingT, MetricsT, ModelT](
        dataset: DataT,
        training: TrainingT,
        metrics: MetricsT,
        model: ModelT,
        **kwargs
    ) -> Config[DataT, TrainingT, MetricsT, ModelT]:
        return Config[DataT, TrainingT, MetricsT, ModelT](
            dataset = dataset,
            training = training,
            metrics = metrics,
            model = model,
            **kwargs
        )
    
    @staticmethod
    def from_args(kwargs: dict)-> Config:
        dataset_data: dict | DatasetConfig = kwargs["dataset"]
        if isinstance(dataset_data, dict):
            dataset_type: str = "default"
            if "type" in dataset_data:
                dataset_type = dataset_data.pop("type")
            
            kwargs["dataset"] = (
                DatasetConfigFactory()
                .create_dataset_config(dataset_type, **dataset_data)
            )
        
        training_data: dict | TrainingConfig = kwargs["training"]
        if isinstance(training_data, dict):
            training_type: str = "default"
            if "type" in training_data:
                training_type = training_data.pop("type")
            
            kwargs["training"] = (
                TrainingConfigFactory()
                .create_training_config(training_type, **training_data)
            )
        
        if "metrics" in kwargs:
            metrics_data: dict | MetricsConfig = kwargs["metrics"]
            if isinstance(metrics_data, dict):
                metrics_type: str = "default"
                if "type" in metrics_data:
                    metrics_type = metrics_data.pop("type")
                
                kwargs["metrics"] = (
                    MetricsConfigFactory()
                    .create_metrics_config(metrics_type, **metrics_data)
                )
        else:
            kwargs["metrics"] = MetricsConfigFactory().create_metrics_config("default")
        
        model_data: dict | ModelConfig = kwargs["model"]
        if isinstance(model_data, dict):
            model_type: str = model_data.get("type", "default")
            kwargs["model"] = (
                ModelConfigFactory().create_model_config(model_type, **model_data)
            )

        return ConfigFactory.from_fields(**kwargs)
