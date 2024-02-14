

from iapytoo.utils.meta_singleton import MetaSingleton

class ModelError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class ModelFactory(metaclass=MetaSingleton):

    def __init__(self) -> None:
        self.models_dict = {}

    def register_model(self, key, model_cls):
        self.models_dict[key] = model_cls

    def create_model(self, config: dict, loader, device='cpu'):
        kind = config["type"]
        try:
            model = self.models_dict[kind](loader, config)
        except KeyError:
            raise ModelError(f"model {kind} is not handled")

        model = model.to(device=device)
        return model
