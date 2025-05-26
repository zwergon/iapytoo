from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class Valuator(ABC):

    def __init__(self, device='cpu'):
        self.device = device

    @abstractmethod
    def evaluate_one(self, input):
        pass

    @abstractmethod
    def evaluate_loader(self, loader: DataLoader):
        pass


class ModelValuator(Valuator):

    def __init__(self, model: nn.Module, device='cpu'):
        super().__init__(device=device)
        self.model = model
        self.model.to(device)

    def evaluate_one(self, input):

        self.model.eval()
        with torch.no_grad():
            return self.model(input)

    def evaluate_loader(self, loader: DataLoader):
        self.model.eval()
        with torch.no_grad():
            for X, Y in loader:
                X = X.to(self.device)
                model_output = self.model(X)
                outputs = model_output.detach().cpu()
                actual = Y.detach().cpu()
                yield outputs, actual


class WGANValuator(ModelValuator):

    # override
    def evaluate_loader(self, loader: DataLoader):

        device = self.device
        generator = self.model
        generator.eval()
        with torch.no_grad():
            for X in loader:
                X = X.to(device)
                gen_output = generator(X)

                outputs = gen_output.detach().cpu()

                yield outputs, None
