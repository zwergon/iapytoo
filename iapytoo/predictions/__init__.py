import torch
import numpy as np
import numbers

from iapytoo.predictions.plotters import PredictionPlotter


class Predictions:
    def __init__(
        self, loader, norm=False, prediction_plotter: PredictionPlotter = None
    ) -> None:
        self.loader = loader
        dataset = self.loader.dataset
        self.norm = norm
        _, Y = dataset[0]
        if isinstance(Y, numbers.Real):
            y_shape = (1,)
        else:
            y_shape = Y.shape
        shape = (len(dataset),) + y_shape
        self.predicted = np.zeros(shape=shape)
        self.actual = np.zeros(shape=shape)

        self.prediction_plotter = self.__attach_plotter(prediction_plotter)

    def __attach_plotter(self, prediction_plotter: PredictionPlotter):
        if prediction_plotter is not None:
            prediction_plotter.predictions = self

        return prediction_plotter

    def compute(self, training):
        model = training.model
        device = training.device
        y_scaling = training.y_scaling

        idx = 0
        model.eval()
        with torch.no_grad():
            for X, Y in self.loader:
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model.predict(X)

                for i in range(Y.shape[0]):
                    predicted = Y_hat[i].detach().cpu()
                    actual = Y[i].detach().cpu()

                    if not self.norm and y_scaling is not None:
                        predicted = y_scaling.inv(predicted)
                        actual = y_scaling.inv(actual)

                    self.predicted[idx] = predicted
                    self.actual[idx] = actual
                    idx = idx + 1

    def plot(self, epoch):
        if self.prediction_plotter is not None:
            return self.prediction_plotter.plot(epoch)
        else:
            return None, None
