import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from iapytoo.dataset.scaling import Scaling
from iapytoo.predictions import PredictionPlotter
from iapytoo.train.models import Model, ModelFactory
from iapytoo.utils.config import Config
from iapytoo.train.training import Training

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class MnistModel(Model):
    def __init__(self, loader, config):
        super(MnistModel, self).__init__(loader, config)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def predict(self, X):
        output = self.forward(X)
        return torch.argmax(output, dim=1)


class ConfusionPlotter(PredictionPlotter):
    def __init__(self):
        super().__init__()

    def plot(self, epoch):
        # Calcul de la matrice de confusion
        cm = confusion_matrix(self.predictions.predicted, self.predictions.actual)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(cm)

        return "confusion_matrix", fig


class MnistTraining(Training):
    def __init__(self, config: Config) -> None:
        super().__init__(config, [], ConfusionPlotter(), None)


if __name__ == "__main__":
    import os

    config_name = os.path.join(os.path.dirname(__file__), "config_mnist.json")
    config = Config(config_name)

    ModelFactory().register_model("mnist", MnistModel)

    training = MnistTraining(config)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=config.batch_size, num_workers=config.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=config.batch_size, num_workers=config.num_workers
    )

    training.fit(train_loader=train_loader, valid_loader=test_loader)
