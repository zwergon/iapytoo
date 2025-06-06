from __future__ import annotations
import torch

from .types import PredictionType
from .plotters import CollectionPlotters

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iapytoo.train.inference import Inference


class Predictions:
    def __init__(self, inference: Inference):
        self.outputs = None  # dimension is assigned in compute
        self.actual = None

        self.inference = inference

        self.prediction_plotter: CollectionPlotters = CollectionPlotters()
        self.prediction_plotter.connect(self)

    @property
    def valuator(self):
        assert self.inference is not None, "no inference for this prediction"
        return self.inference.valuator

    @property
    def predictor(self):
        assert self.inference is not None, "no inference for this prediction"
        return self.inference.predictor

    def __len__(self):
        """return the number of plotters used by the predictions"""
        return len(self.prediction_plotter)

    def add_plotter(self, prediction_plotter):
        self.prediction_plotter.add(prediction_plotter)

    def compute(self, loader):

        assert self.valuator is not None, "no valuator"
        assert self.predictor is not None, "no predictor"

        self.outputs = torch.zeros(size=(0,))
        self.actual = torch.zeros(size=(0,))

        for outputs, actual in self.valuator.evaluate_loader(loader):
            self.outputs = torch.cat((self.outputs, outputs), dim=0)

            if actual is not None:
                self.actual = torch.cat((self.actual, actual), dim=0)

    def tensor(self, type: PredictionType = PredictionType.PREDICTED):
        if type == PredictionType.PREDICTED:
            return self.predictor(self.outputs)
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
