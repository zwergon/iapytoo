import os
import sys
import logging
import mlflow
import tempfile
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union

os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


class DatasetConfig(BaseModel):
    normalization: Optional[bool] = False
    batch_size: int
    indices: List[int] = [0]
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
    weight_decay: Optional[float] = 1e-4
    momentum: Optional[float] = 0.9
    scheduler: Optional[str] = "step"
    step_size: Optional[int] = 10
    gamma: Optional[float] = 0.9
    lambda_gp: Optional[float] = 10.
    groups: Optional[int] = 1
    top_accuracy: Optional[int] = 3 

class ModelConfig(BaseModel):
    type: str = "default"
    model: str
    hidden_size: Optional[int] = 128
    num_layers: Optional[int] = 3
    kernel_size: Optional[int] = 5
    dropout: Optional[float] = 0.5

class GanConfig(ModelConfig):
    generator: str
    discriminator: str
    
class Config(BaseModel):
    project: str
    run: str
    tracking_uri: Optional[str] = None
    sensors: Optional[str] = None
    cuda: Optional[bool] = True
    seed: Optional[int] = 42
    dataset: DatasetConfig
    training: TrainingConfig
    model: Union[ModelConfig, GanConfig]


    def to_flat_dict(self) -> Dict[str, str]:
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

        return flatten(self.model_dump())

    @classmethod
    def create_from_args(cls, args: dict):
        
        model_data = args.pop("model", {})  # Enlève "model" du dictionnaire principal
        model_type = model_data.get("type", "default")

        if model_type == "GAN":
            model_instance = GanConfig(**model_data)
        else:
            model_instance = ModelConfig(**model_data)

        return cls(**args, model=model_instance)  # On passe un objet instancié
    
    @classmethod
    def create_from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        return cls.create_from_args(data)

    @staticmethod
    def _indices(vars):
        v_list = vars[1:-1].split(",")
        return [int(v) for v in v_list]

    @staticmethod
    def _bool(var):
        if var == "True":
            return True
        else:
            return False

    @classmethod
    def create_from_run_id(cls, run_id, tracking_uri=None):
        cf = {}

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        run = mlflow.get_run(run_id)

        nested_dict = {}
        for key, value in run.data.params.items():
            keys = key.split(".")
            d = nested_dict
            for key in keys[:-1]:
                if key not in d:   
                    d[key] = {}
                d = d[key]  
            d[keys[-1]] = value  # TODO May check and cast type again !

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
        return self.model.type.lower() == "gan"