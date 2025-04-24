import torch
import numpy as np

from iapytoo.utils.config import Config
from .types import PredictionType
from .plotters import CollectionPlotters
from .predictors import Predictor, PredictorFactory


class Valuator:

    def __init__(self, loader, device="cpu"):
        self.loader = loader
        self.device = device

    def evaluate(self):
        pass


class Predictions:
    def __init__(self, config: Config):
        self.outputs = None  # dimension is assigned in compute
        self.actual = None

        self.predictor = PredictorFactory().create_predictor(config)

        self.prediction_plotter: CollectionPlotters = CollectionPlotters()
        self.prediction_plotter.connect(self)

    def __len__(self):
        """return the number of plotters used by the predictions"""
        return len(self.prediction_plotter)

    def add_plotter(self, prediction_plotter):
        self.prediction_plotter.add(prediction_plotter)

    def compute(self, valuator: Valuator):

        self.outputs = torch.zeros(size=(0,))
        self.actual = torch.zeros(size=(0,))

        for outputs, actual in valuator.evaluate():
            self.outputs = torch.cat((self.outputs, outputs), dim=0)

            if actual is not None:
                self.actual = torch.cat((self.actual, actual), dim=0)

    def tensor(self, type: PredictionType = PredictionType.PREDICTED):
        if type == PredictionType.PREDICTED:
            return self.predictor(self.outputs) if self.predictor else self.outputs
        elif type == PredictionType.ACTUAL:
            return self.actual
        else:
            return self.outputs

    def numpy(self, type: PredictionType = PredictionType.PREDICTED):
        return self.tensor(type).numpy()

    def plot(self, epoch):
        if self.prediction_plotter is not None:
            return self.prediction_plotter.plot(epoch)
        else:
            return {}  # nothing to plot
