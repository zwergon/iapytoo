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
