import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR


from iapytoo.dataset.scaling import Scaling
from iapytoo.predictions.predictors import MaxPredictor
from iapytoo.predictions.plotters import ConfusionPlotter
from iapytoo.train.factories import Model, ModelFactory, SchedulerFactory, Scheduler
from iapytoo.utils.config import ConfigFactory, Config
from iapytoo.train.training import Training
from iapytoo.train.mlflow_model import Transform


import matplotlib.pyplot as plt

from mlflow.types.schema import TensorSpec, Schema
from mlflow.models import ModelSignature


class MnistModel(Model):
    def __init__(self, loader, config: Config):
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

    def predict(self, model_output):
        return torch.argmax(model_output, dim=1)


class MnistScheduler(Scheduler):
    def __init__(self, optimizer, config: Config) -> None:
        super().__init__(optimizer, config)

        def lr_lambda(epoch):
            # LR to be 0.1 * (1/1+0.01*epoch)
            return 0.995**epoch

        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


class MnistTraining(Training):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ConfusionPlotter())


class MnistTransform(Transform):

    def __init__(self, dataset):
        super().__init__(
            train_transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            ),
            dataset=dataset
        )

    # override
    def set_signature(self, dataset):
        img, Y = dataset[0]
        X = Transform.img_to_numpy(img)
        # add batch size
        X = np.expand_dims(X, axis=0)
        Y = np.expand_dims(Y, axis=0)
        x_shape = list(X.shape)
        x_shape[0] = -1
        y_shape = list(Y.shape)
        y_shape[0] = -1

        input_schema = Schema(
            [TensorSpec(type=X.dtype, shape=x_shape)])
        output_schema = Schema(
            [TensorSpec(type=Y.dtype, shape=y_shape)])

        self.signature = ModelSignature(
            inputs=input_schema, outputs=output_schema)
        self.input_example = X

    # override

    def transform(self, context, model_input, params=None):

        model_input_transformed = [
            self.infer_transform(img) for img in model_input]
        print(model_input_transformed[0].permute(1, 0, 2).shape)

        model_input_tensor = torch.stack(
            [img.permute(1, 0, 2) for img in model_input_transformed])

        return model_input_tensor


if __name__ == "__main__":
    from iapytoo.utils.arguments import parse_args

    ModelFactory().register_model("mnist", MnistModel)
    SchedulerFactory().register_scheduler("mnist", MnistScheduler)

    args = parse_args()

    if args.run_id is not None:
        config = ConfigFactory.from_run_id(args.run_id, args.tracking_uri)
        config.training.epochs = args.epochs
    else:
        # INPUT Parameters
        config = ConfigFactory.from_yaml(args.yaml)

    Training.seed(config)

    signature_dataset = datasets.MNIST(
        config.dataset.path,
        train=False
    )
    transform = MnistTransform(signature_dataset)

    training = MnistTraining(config)
    train_dataset = datasets.MNIST(
        config.dataset.path,
        train=True,
        download=True,
        transform=transform.train_transform
    )

    train_dataset = datasets.MNIST(
        config.dataset.path,
        train=False,
        transform=transform.infer_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )
    test_loader = DataLoader(
        train_dataset, batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers
    )

    training.fit(
        train_loader=train_loader,
        valid_loader=test_loader,
        transform=transform,
        run_id=args.run_id
    )
