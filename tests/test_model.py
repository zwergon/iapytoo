
from iapytoo.train.models import ModelFactory

if __name__ == "__main__":
    factory = ModelFactory()
    print(factory)
    factory.register_model("test", 3)


    print(ModelFactory().models_dict['test'])

