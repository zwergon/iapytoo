import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


from iapytoo.dataset.transform import MeanNormalize
from iapytoo.train.factories import Model, Scheduler
from iapytoo.utils.config import Config
from iapytoo.train.mlflow_model import MlflowTransform, IMlfowModelProvider


class MnistModel(Model):
    def __init__(self, config: Config):
        super(MnistModel, self).__init__(config)
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
        super().__init__(
            transform=MeanNormalize(0.1307, 0.3081))

    # override
    def __call__(self, model_input, *args, **kwds) -> np.array:
        return self.transform(model_input)


class MnistMlfowModel(IMlfowModelProvider):

    def __init__(self):
        X = np.random.rand(1, 28, 28)
        Y = np.zeros(shape=(1,), dtype=np.int64)
        self.input_example = X
        self.transform: MlflowTransform = MnistTransform()

    # override
    def code_definition(self) -> dict:
        from pathlib import Path
        return {
            "path": str(Path(__file__).parent),
            "module": "examples.subclasses",
            "model_cls": "MnistModel",
            "transform_cls": "MnistTransform"
        }

    # override

    def get_transform(self) -> MlflowTransform:
        return self.transform

    # override
    def get_input_example(self) -> np.array:
        return self.input_example
