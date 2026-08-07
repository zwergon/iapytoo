from pydantic import BaseModel
from typing import Optional, Union
from iapytoo.utils.singleton import singleton


class ModelConfig(BaseModel):
    type: str
    provider: str
    # Backbone pluggable au runtime (cf. Factory.register_backbone/create_backbone) et
    # dimension de conditionnement partagee par les architectures conditionnelles ; les deux
    # restent optionnels pour ne rien changer aux modeles qui n'utilisent pas ce pattern.
    backbone: Optional[str] = None
    cond_dim: Optional[int] = None
    # Colonnes physiques (indices dans conditions/input_values) et leurs labels
    # utilises comme conditionnement (c), dans cet ordre -- permet a un
    # predict() rechargeant uniquement l'artefact MLflow de savoir quelles
    # cles de conditionnement il attend (cf. iapytoo/mlflow/model.py, MlInput.
    # condition). Optionnels, sans impact sur les modeles non conditionnels.
    cond_indices: Optional[list[int]] = None
    cond_labels: Optional[list[str]] = None
    n_channels: Optional[int] = 3
    base: Optional[int] = 64
    t_dim: Optional[int] = 256


class DefaultModelConfig(ModelConfig):
    hidden_size: Optional[int] = 128
    num_layers: Optional[int] = 3
    kernel_size: Optional[int] = 5
    dropout: Optional[float] = 0.5


class GanConfig(ModelConfig):
    hidden_size: Optional[int] = 128
    lambda_gp: Optional[float] = 10.0
    noise_dim: Optional[int] = 100
    signal_length: Optional[int] = 200
    n_critic: Optional[int] = 5


class DDPMConfig(ModelConfig):
    lambda_: Optional[float] = 0.1
    n_times: Optional[int] = 1000
    signal_length: Optional[int] = 512
    # Bornes du schedule de bruit (torch.linspace(beta_start, beta_end, n_times)).
    # Les valeurs par defaut (1e-4, 0.02) sont celles du papier DDPM original,
    # calibrees pour n_times=1000 : alphas_cumprod[n_times-1] y est quasi nul
    # (~4e-5), ce qui rend le point de depart de l'echantillonnage (bruit gaussien
    # pur) coherent avec ce que le modele voit a l'entrainement au pas le plus
    # bruite. Avec un n_times plus petit et ces memes bornes, ce schedule ne
    # sature plus (ex. n_times=100 -> alphas_cumprod[-1]~0.36, 60% de signal
    # residuel au pas le plus bruite) : le modele n'a alors jamais vu, a
    # l'entrainement, une entree aussi peu bruitee que le vrai bruit pur duquel
    # part la generation - mismatch train/inference degradant fortement la
    # qualite. Si n_times est reduit, augmenter beta_end en consequence pour
    # re-saturer le schedule (regle simple : beta_end ~ 0.02 * 1000/n_times
    # garde alphas_cumprod[-1] du meme ordre de grandeur qu'a n_times=1000).
    beta_start: Optional[float] = 1e-4
    beta_end: Optional[float] = 0.02


class MLFlowConfig(ModelConfig):
    run_id: str


class ConfigError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


@singleton
class ModelConfigFactory:
    def __init__(self) -> None:
        self.model_dict = {
            "default": DefaultModelConfig,
            "mlflow": MLFlowConfig,
            "gan": GanConfig,
            "ddpm": DDPMConfig
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
