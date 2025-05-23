import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image


from iapytoo.predictions.plotters import ConfusionPlotter
from iapytoo.train.factories import Model, Scheduler, Factory
from iapytoo.utils.config import ConfigFactory, Config
from iapytoo.train.training import Training
from iapytoo.train.mlflow_model import MlflowTransform, IMlfowModelProvider


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


class MnistTransform(MlflowTransform):

    def __init__(self) -> None:
        super().__init__(transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        ))

    # override
    def __call__(self, model_input, *args, **kwds):

        images_pil = []
        for i in range(model_input.shape[0]):
            # Extraire l'image (en supprimant la dimension du canal si nécessaire)
            # Supposons que l'image est en niveaux de gris
            img_array = model_input[i, 0, :, :]

            # Normaliser l'image si nécessaire (par exemple, si les valeurs sont entre 0 et 1)
            img_array = (img_array * 255).astype(np.uint8)

            # Créer une image PIL
            # 'L' pour les niveaux de gris
            img_pil = Image.fromarray(img_array, 'L')
            images_pil.append(img_pil)

        model_input_tensor = torch.stack(
            [self.transform(img) for img in images_pil])

        return model_input_tensor


class MnistMlfowModel(IMlfowModelProvider):

    def __init__(self):
        X = np.random.rand(1, 1, 28, 28)
        Y = np.zeros(shape=(1,), dtype=np.int64)
        # add batch size
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
        self.transform: MlflowTransform = MnistTransform()

    # override
    def get_transform(self) -> MlflowTransform:
        return self.transform

    # override
    def get_signature(self):
        return self.signature

    # override
    def get_input_example(self):
        return self.input_example


class MnistTraining(Training):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.predictions.add_plotter(ConfusionPlotter())

        self.mlflow_model_provider: IMlfowModelProvider = MnistMlfowModel()

    @property
    def mlflow_transform(self):
        return training.mlflow_model_provider.get_transform()


if __name__ == "__main__":
    from iapytoo.utils.arguments import parse_args

    factory = Factory()
    factory.register_model("mnist", MnistModel)
    factory.register_scheduler("mnist", MnistScheduler)

    args = parse_args()

    if args.run_id is not None:
        config = ConfigFactory.from_run_id(args.run_id, args.tracking_uri)
        config.training.epochs = args.epochs
    else:
        # INPUT Parameters
        config = ConfigFactory.from_yaml(args.yaml)

    Training.seed(config)

    training = MnistTraining(config)
    mflow_transform: MlflowTransform = training.mlflow_transform
    train_dataset = datasets.MNIST(
        config.dataset.path,
        train=True,
        download=True,
        transform=mflow_transform.transform
    )

    test_dataset = datasets.MNIST(
        config.dataset.path,
        train=False,
        transform=mflow_transform.transform
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
        run_id=args.run_id
    )
