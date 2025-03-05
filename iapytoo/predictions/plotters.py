import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.signal import welch
import numpy as np

from iapytoo.predictions.types import PredictionType


class PredictionPlotter:
    def __init__(self, title=None):
        self.predictions = None
        self.title = title

    def connect(self, predictions):
        self.predictions = predictions

    def plot(self, epoch):
        return None


class CollectionPlotters(PredictionPlotter):

    def __init__(self, title=None):
        super(CollectionPlotters, self).__init__(title)
        self.plotters = []

    def __len__(self):
        return len(self.plotters)

    def add(self, prediction_plotter):
        prediction_plotter.connect(self.predictions)
        self.plotters.append(prediction_plotter)

    def plot(self, epoch):
        plots = {}
        for p in self.plotters:
            plots.update(p.plot(epoch))

        return plots


class ScatterPlotter(PredictionPlotter):
    def __init__(self):
        super().__init__(title="actual_versus_predicted")

    def plot(self, epoch):
        predicted = self.predictions.numpy(PredictionType.PREDICTED)
        actual = self.predictions.numpy(PredictionType.ACTUAL)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(predicted, actual)

        return {self.title: fig}


class ConfusionPlotter(PredictionPlotter):
    def __init__(self):
        super().__init__(title="confusion_matrix")

    def plot(self, epoch):
        # Calcul de la matrice de confusion
        predicted = self.predictions.numpy(PredictionType.PREDICTED)
        actual = self.predictions.numpy(PredictionType.ACTUAL)
        cm = confusion_matrix(predicted, actual)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

        return {self.title: fig}


class TSNEPlotter(PredictionPlotter):

    def __init__(self):
        super().__init__(title="tsne")

    def plot(self, epoch):

        outputs = self.predictions.numpy(PredictionType.OUTPUTS)
        actual = self.predictions.numpy(PredictionType.ACTUAL).astype(np.int32)
        tsne = TSNE(n_components=2, random_state=42)
        predicted_tsne = tsne.fit_transform(outputs)

        fig, ax = plt.subplots(figsize=(10, 5))
        # Utilisons les classes pour colorer les points
        for i in np.unique(actual):
            ax.scatter(
                predicted_tsne[actual == i, 0],
                predicted_tsne[actual == i, 1],
                label=f"{i}",
                alpha=0.6,
            )
        ax.legend()
        ax.set_title("t-SNE")

        return {self.title: fig}


class Fake1DPlotter(PredictionPlotter):
    def __init__(self, n_plot=4):
        super().__init__(title="generated_fake1d")
        self.n_plot = n_plot

    def plot(self, epoch):
        fake = self.predictions.numpy(PredictionType.PREDICTED)
        f, a = plt.subplots(self.n_plot, self.n_plot, figsize=(8, 8))
        for i in range(self.n_plot):
            for j in range(self.n_plot):
                a[i][j].plot(fake[i * self.n_plot + j, 0, :])
                a[i][j].set_xticks(())
                a[i][j].set_yticks(())

        return {self.title: f}


class DSPPlotter(PredictionPlotter):
    def __init__(self, f_max=4, nperseg=200):
        super().__init__(title="dsp")
        self.f_max = f_max
        self.nperseg = nperseg

    def plot(self, epoch):
        fake = self.predictions.numpy(PredictionType.PREDICTED)
        assert fake is not None and fake.shape[0] > 0, "no signal for DSP extraction"
        f, ax = plt.subplots()

        frequencies, power_spectrum = welch(
            fake[0, 0, :], fs=self.f_max, nperseg=self.nperseg)
        ax.plot(frequencies, power_spectrum)

        return {self.title: f}


class Fake2DPlotter(PredictionPlotter):
    def __init__(self, n_plot=4):
        super().__init__("generated_fake2d")
        self.n_plot = n_plot

    def plot(self, epoch):
        outputs = self.predictions.tensor(PredictionType.OUTPUTS)
        fake_list = [outputs[i] for i in range(outputs.shape[0])]
        grid_img = torchvision.utils.make_grid(fake_list, nrow=self.n_plot)

        f, ax = plt.subplots()
        ax.imshow(grid_img.permute(1, 2, 0))

        return {self.title: f}
