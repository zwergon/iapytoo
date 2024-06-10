import torchvision
import matplotlib.pyplot as plt


class PredictionPlotter:
    def __init__(self):
        self.predictions = None

    def plot(self, epoch):
        return None, None


class ScatterPlotter(PredictionPlotter):
    def __init__(self):
        super().__init__()

    def plot(self, epoch):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.predictions.predicted, self.predictions.actual)

        return "actual_versus_predicted", fig


class Fake1DPlotter(PredictionPlotter):
    def __init__(self, n_plot=4):
        super().__init__()
        self.n_plot = n_plot

    def plot(self, epocht):
        fake = self.predictions.predicted
        f, a = plt.subplots(self.n_plot, self.n_plot, figsize=(8, 8))
        for i in range(self.n_plot):
            for j in range(self.n_plot):
                a[i][j].plot(fake[i * self.n_plot + j].view(-1))
                a[i][j].set_xticks(())
                a[i][j].set_yticks(())

        return "Generated", f


class Fake2DPlotter(PredictionPlotter):
    def __init__(self, n_plot=4):
        super().__init__()
        self.n_plot = n_plot

    def plot(self, epoch):
        fake_list = self.predictions.predicted
        grid_img = torchvision.utils.make_grid(fake_list, nrow=self.n_plot)

        f, ax = plt.subplots()
        ax.imshow(grid_img.permute(1, 2, 0))

        return "Generated", f
