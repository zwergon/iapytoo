import torchvision
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.signal import welch


class PredictionPlotter:
    def __init__(self, title=None):
        self.predictions = None
        self.title = title

    def plot(self, epoch):
        return None


class CollectionPlotters(PredictionPlotter):

    def __init__(self, title=None):
        super(CollectionPlotters, self).__init__(title)
        self.plotters = []

    def plot(self, epoch):
        plots = {}
        for p in self.plotters:
            plots.append(p.plot(epoch))

        return plots

class ScatterPlotter(PredictionPlotter):
    def __init__(self):
        super().__init__(title="actual_versus_predicted")

    def plot(self, epoch):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.predictions.predicted, self.predictions.actual)

        return {self.title: fig}


class ConfusionPlotter(PredictionPlotter):
    def __init__(self):
        super().__init__(title="confusion_matrix")

    def plot(self, epoch):
        # Calcul de la matrice de confusion
        cm = confusion_matrix(self.predictions.predicted, self.predictions.actual)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

        return {self.title: fig}


class Fake1DPlotter(PredictionPlotter):
    def __init__(self, n_plot=4):
        super().__init__(title="generated_fake1d")
        self.n_plot = n_plot

    def plot(self, epocht):
        fake = self.predictions.predicted
        f, a = plt.subplots(self.n_plot, self.n_plot, figsize=(8, 8))
        for i in range(self.n_plot):
            for j in range(self.n_plot):
                a[i][j].plot(fake[i * self.n_plot + j].view(-1))
                a[i][j].set_xticks(())
                a[i][j].set_yticks(())

        return  {self.title: f}


class DSPPlotter(PredictionPlotter):
    def __init__(self, f_max=4, nperseg=200 ):
        super().__init__(title="dsp")
        self.f_max = f_max
        self.nperseg = nperseg

    def plot(self, epoch):
        fake = self.predictions.predicted[0]
        f, ax = plt.subplots()
        
        frequencies, power_spectrum = welch(fake, fs=self.f_max, nperseg=self.nperseg)  
        ax.plot(frequencies, power_spectrum)   

        return  {self.title: f}



class Fake2DPlotter(PredictionPlotter):
    def __init__(self, n_plot=4):
        super().__init__("generated_fake2d")
        self.n_plot = n_plot

    def plot(self, epoch):
        fake_list = self.predictions.predicted
        grid_img = torchvision.utils.make_grid(fake_list, nrow=self.n_plot)

        f, ax = plt.subplots()
        ax.imshow(grid_img.permute(1, 2, 0))

        return {self.title: f}
