import torch
import numpy as np


class PredictionPlotter:

    def __init__(self):
        self.predictions = None

    def plot(self, epoch):
        return None, None

class Predictions:
    def __init__(self, loader, norm=False, prediction_plotter: PredictionPlotter = None) -> None:
        self.loader = loader
        dataset = self.loader.dataset
        self.norm = norm
        _, Y = dataset[0]
        shape =  (len(dataset),) + Y.shape
        self.predicted = np.zeros(shape=shape)
        self.actual = np.zeros(shape=shape)

    def compute(self, training):
        
        model = training.model
        device = training.device

        idx = 0
        model.eval()
        with torch.no_grad():
            for X, Y in self.loader:
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model(X)

                for i in range(Y.shape[0]):
                    predicted = Y_hat[i].detach().cpu()
                    actual = Y[i].detach().cpu()
                    
                    # TODO add normalization
                    # dataset: WBDataset = self.loader.dataset
                    # if not self.norm and dataset.norma:
                    #     dataset.norma.unnorm_y(predicted)
                    #     dataset.norma.unnorm_y(actual)

                    self.predicted[idx] = predicted
                    self.actual[idx] = actual
                    idx = idx + 1

    def plot(self, epoch):
        if self.prediction_plotter is not None:
            return self.prediction_plotter.plot(epoch)
        else:
            return None, None

