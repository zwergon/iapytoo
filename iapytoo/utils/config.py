import os
import sys
import logging
import mlflow
import yaml
from pydantic import BaseModel, BeforeValidator, Field
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
    ratio_train_test: Optional[float] = 0.8
    num_workers: Optional[int] = 2


class TrainingConfig(BaseModel):
    epochs: int = 10
    tqdm: Optional[bool] = True
    n_steps_by_batch: Optional[int] = 10
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


class Config(BaseModel):
    project: str
    run: str
    tracking_uri: Optional[str] = None
    sensors: Optional[str] = None
    cuda: Optional[bool] = True
    seed: Optional[int] = 42
    dataset: DatasetConfig
    training: Optional[TrainingConfig] = None
    metrics: MetricsConfig = Field(default=MetricsConfig())
    model: ModelConfig

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

        for key, value in run.data.params.items():
            if isinstance(value, str) and value == 'None':
                continue
            keys = key.split(".")
            d = nested_dict
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
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
