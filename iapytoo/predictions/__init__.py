import torch
import numpy as np
import numbers

from iapytoo.predictions.plotters import PredictionPlotter


class Predictions:
    def __init__(
        self, loader, norm=False, prediction_plotter: PredictionPlotter = None
    ):
        self.loader = loader
        self.norm = norm
        self.predicted = []
        self.actual = []

        self.prediction_plotter = self.__attach_plotter(prediction_plotter)

    def __attach_plotter(self, prediction_plotter: PredictionPlotter):
        if prediction_plotter is not None:
            prediction_plotter.predictions = self

        return prediction_plotter

    def compute(self, training):
        device = training.device
        y_scaling = training.y_scaling
        self.predicted = []
        self.actual = []

        model = training.model  # first model is generator in gan !
        model.eval()
        with torch.no_grad():
            for X, Y in self.loader:
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model.predict(X)
                for i in range(Y_hat.shape[0]):
                    predicted = Y_hat[i].detach().cpu()
                    actual = Y[i].detach().cpu()

                    if not self.norm and y_scaling is not None:
                        predicted = y_scaling.inv(predicted)
                        actual = y_scaling.inv(actual)

                    self.predicted.append(predicted)
                    self.actual.append(actual)

    def plot(self, epoch):
        if self.prediction_plotter is not None:
            return self.prediction_plotter.plot(epoch)
        else:
            return {} # nothing to plot


class GenerativePredictions(Predictions):
    def __init__(
        self, loader, norm=False, prediction_plotter: PredictionPlotter = None
    ):
        super().__init__(loader, norm, prediction_plotter)

    def compute(self, training):
        self.predicted = []
        device = training.device
        generator = training.generator
        generator.eval()
        with torch.no_grad():
            for X in self.loader:
                X = X.to(device)
                Y_hat = generator.predict(X)
                for i in range(Y_hat.shape[0]):
                    predicted = Y_hat[i].detach().cpu()
                    self.predicted.append(predicted)
