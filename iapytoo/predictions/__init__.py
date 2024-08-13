import torch
import numpy as np

from .types import PredictionType
from .plotters import CollectionPlotters
from .predictors import Predictor


class Predictions:
    def __init__(self, predictor: Predictor, norm=False):
        self.norm = norm
        self.outputs = None  # dimension is assigned in compute
        self.actual = None
        self.predictor = predictor
        self.prediction_plotter: CollectionPlotters = CollectionPlotters()
        self.prediction_plotter.connect(self)

    def __len__(self):
        """return the number of plotters used by the predictions"""
        return len(self.prediction_plotter)

    def add_plotter(self, prediction_plotter):
        self.prediction_plotter.add(prediction_plotter)

    def compute(self, training, loader):
        device = training.device
        y_scaling = training.y_scaling
        self.outputs = torch.zeros(size=(0,), device=device)
        self.actual = torch.zeros(size=(0,), device=device)

        model = training.model
        model.eval()
        with torch.no_grad():
            for X, Y in loader:
                X = X.to(device)
                Y = Y.to(device)
                model_output = model(X)

                self.outputs = torch.cat((self.outputs, model_output), dim=0)
                self.actual = torch.cat((self.actual, Y), dim=0)

        if not self.norm and y_scaling is not None:
            self.outputs = y_scaling.inv(self.outputs)
            self.actual = y_scaling.inv(self.actual)

    def tensor(self, type: PredictionType = PredictionType.PREDICTED):
        if type == PredictionType.PREDICTED:
            predicted = self.predictor(self.outputs)
            return predicted.detach().cpu()
        elif type == PredictionType.ACTUAL:
            return self.actual.detach().cpu()
        else:
            return self.outputs.detach().cpu()

    def numpy(self, type: PredictionType = PredictionType.PREDICTED):
        return self.tensor(type).numpy()

    def plot(self, epoch):
        if self.prediction_plotter is not None:
            return self.prediction_plotter.plot(epoch)
        else:
            return {}  # nothing to plot


class GenerativePredictions(Predictions):
    def __init__(self, norm=False):
        super().__init__(norm)

    def compute(self, training, loader):
        device = training.device
        self.outputs = torch.zeros(size=(0,), device=device)
        generator = training.generator
        generator.eval()
        with torch.no_grad():
            for X in loader:
                X = X.to(device)
                gen_output = generator(X)
                self.outputs = torch.cat((self.outputs, gen_output), dim=0)
