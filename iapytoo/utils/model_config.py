
from pydantic import BaseModel
from typing import Optional, Dict, Type, Union, Literal
from iapytoo.utils.singleton import singleton


class ModelConfig(BaseModel):
    type: str
    predictor: Optional[str] = None

    def _network(self) -> str:
        pass


class DefaultModelConfig(ModelConfig):
    model: str
    hidden_size: Optional[int] = 128
    num_layers: Optional[int] = 3
    kernel_size: Optional[int] = 5
    dropout: Optional[float] = 0.5

    def _network(self) -> str:
        return str(self.model)


class GanConfig(ModelConfig):
    generator: str
    discriminator: str
    lambda_gp: Optional[float] = 10.
    noise_dim: Optional[int] = 100
    n_critic: Optional[int] = 5

    def _network(self) -> str:
        return f"{self.discriminator} & {self.generator}"


class MLFlowConfig(ModelConfig):
    run_id: str

    def _network(self) -> str:
        return self.run_id


class ConfigError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


@singleton
class ModelConfigFactory:
    def __init__(self) -> None:
        self.model_dict = {
            "default": DefaultModelConfig,
            "mlflow": MLFlowConfig,
            "gan": GanConfig
        }
    
    def register_model_config(
        self, key: str, model_config_cls: type[ModelConfig]
    ) -> None:
        self.model_dict[key] = model_config_cls

    def get_union_type(self):
        return Union[tuple(v for v in self.model_dict.values())]

    def create_model_config(self, kind, **kwargs) -> ModelConfig:

        try:
            model_config: ModelConfig = self.model_dict[kind](**kwargs)
        except KeyError:
            raise ConfigError(f"Config for model {kind} doesn't exist")

        return model_config
