import torch
import numpy as np

class Predictions:
    def __init__(self, loader, norm=False) -> None:
        self.loader = loader
        dataset = self.loader.dataset
        self.norm = norm
        _, Y = dataset[0]
        self.predicted = np.zeros(shape=(len(dataset), Y.shape[0], Y.shape[1]))
        self.actual = np.zeros(shape=(len(dataset), Y.shape[0], Y.shape[1]))

    def compute(self, model, device):
        
        idx = 0
        model.eval()
        with torch.no_grad():
            for X, Y in self.loader:
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model(X)

                for i in range(Y.shape[0]):
                    predicted = Y_hat[i, :, :].detach().cpu()
                    actual = Y[i, :, :].detach().cpu()
                    
                    # TODO add normalization
                    # dataset: WBDataset = self.loader.dataset
                    # if not self.norm and dataset.norma:
                    #     dataset.norma.unnorm_y(predicted)
                    #     dataset.norma.unnorm_y(actual)

                    self.predicted[idx, :, :] = predicted
                    self.actual[idx, :, :] = actual
                    idx = idx + 1
               