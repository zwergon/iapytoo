import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR


from iapytoo.dataset.scaling import Scaling
from iapytoo.predictions import PredictionPlotter
from iapytoo.train.factories import Model, ModelFactory, SchedulerFactory, Scheduler
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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

        return "confusion_matrix", fig


class MnistScheduler(Scheduler):
    def __init__(self, optimizer, config) -> None:
        super().__init__(optimizer, config)

        def lr_lambda(epoch):
            # LR to be 0.1 * (1/1+0.01*epoch)
            return 0.995**epoch

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


class MnistTraining(Training):
    def __init__(self, config: Config) -> None:
        super().__init__(config, [], ConfusionPlotter(), None)


if __name__ == "__main__":
    import os

    from iapytoo.utils.arguments import parse_args

    ModelFactory().register_model("mnist", MnistModel)
    SchedulerFactory().register_scheduler("mnist", MnistScheduler)

    args = parse_args()

    if args.run_id is not None:
        config = Config.create_from_run_id(args.run_id, args.tracking_uri)
        config.epochs = args.epochs
    else:
        # INPUT Parameters
        config = Config.create_from_args(args)

    Training.seed(config)

    config = Config.create_from_args(args)

    training = MnistTraining(config)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = datasets.MNIST(
        args.dataset, train=True, download=True, transform=transform
    )
    dataset2 = datasets.MNIST(args.dataset, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=config.batch_size, num_workers=config.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=config.batch_size, num_workers=config.num_workers
    )

    training.fit(
        train_loader=train_loader, valid_loader=test_loader, run_id=args.run_id
    )
