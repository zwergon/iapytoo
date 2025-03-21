import sys
import random
import numpy
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import logging


from iapytoo.utils.config import Config
from iapytoo.utils.timer import Timer
from iapytoo.utils.iterative_mean import Mean
from iapytoo.dataset.scaling import Scaling
from iapytoo.train.loss import Loss
from iapytoo.train.factories import (
    ModelFactory,
    OptimizerFactory,
    SchedulerFactory,
    LossFactory,
)
from iapytoo.train.logger import Logger
from iapytoo.train.checkpoint import CheckPoint
from iapytoo.predictions import Predictions, Valuator
from iapytoo.metrics import MetricsCollection

from enum import IntEnum


class LossType(IntEnum):
    TRAIN = 0
    VALID = 1


class Inference:

    def __init__(
        self,
        config: Config
    ) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._config = config
        self.logger = None
        self.predictions = Predictions(config)
        self._models = []

    def _display_device(self):
        use_cuda = torch.cuda.is_available()
        if self._config.cuda and use_cuda:
            msg = "\n__CUDA\n"
            msg += f"__CUDNN VERSION: {torch.backends.cudnn.version()}\n"
            msg += f"__Number CUDA Devices: {torch.cuda.device_count()}\n"
            msg += f"__CUDA Device Name: {torch.cuda.get_device_name(0)}\n"
            msg += f"__CUDA Device Total Memory [GB]: {torch.cuda.get_device_properties(0).total_memory / 1e9}\n"
            msg += "-----------\n"
            logging.info(msg)
        else:
            logging.info("__CPU")

    @property
    def model(self):
        return self._models[0]

    def _valuator(self, loader):
        return InferenceValuator(self, loader)

    def _create_models(self, loader):
        pass

    def predict(self, loader, run_id=None):
        pass


class InferenceValuator(Valuator):

    def __init__(self, inference: Inference, loader):
        super().__init__(loader, device=inference.device)
        self.inference = inference

    def evaluate(self):
        model = self.inference.model
        model.eval()
        with torch.no_grad():
            for X, Y in self.loader:
                X = X.to(self.inference.device)
                model_output = model(X)

                outputs = model_output.detach().cpu()
                actual = Y.detach().cpu()
                yield outputs, actual


class Training(Inference):
    @staticmethod
    def seed(config: Config):
        seed = config.seed
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

    def __init__(
        self,
        config: Config
    ) -> None:
        super().__init__(config=config)
        # first init all random seeds
        seed = config.seed
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        self.criterion = self._create_criterion()

        if self._config.training.tqdm:
            self.train_loop = self.__tqdm_loop(self._inner_train)
            self.valid_loop = self.__tqdm_loop(self._inner_validate)
        else:
            self.train_loop = self.__batch_loop(self._inner_train)
            self.valid_loop = self.__batch_loop(self._inner_validate)

        self.loss = Loss(n_losses=2)
        self._optimizers = []
        self._schedulers = []

    @property
    def scheduler(self):
        if len(self._schedulers) > 0:
            return self._schedulers[0].lr_scheduler

        return None

    @property
    def optimizer(self):
        if len(self._optimizers) > 0:
            return self._optimizers[0].torch_optimizer

        return None

    def state_dict(self):
        state_dict = {}
        state_dict["n_models"] = len(self._models)
        for i, m in enumerate(self._models):
            state_dict[f"model_{i}"] = m.state_dict()
        state_dict["n_optimizers"] = len(self._optimizers)
        for i, o in enumerate(self._optimizers):
            state_dict[f"optimizer_{i}"] = o.torch_optimizer.state_dict()
        state_dict["n_schedulers"] = len(self._schedulers)
        for i, s in enumerate(self._schedulers):
            state_dict[f"scheduler_{i}"] = s.lr_scheduler.state_dict()
        state_dict["loss"] = self.loss.state_dict()

        return state_dict

    def load_state_dict(self, state_dict, only_model=False):
        n_models = state_dict["n_models"]
        for i in range(n_models):
            self._models[i].load_state_dict(state_dict[f"model_{i}"])
        if not only_model:
            n_optimizers = state_dict["n_optimizers"]
            for i in range(n_optimizers):
                self._optimizers[i].torch_optimizer.load_state_dict(
                    state_dict[f"optimizer_{i}"]
                )
            n_schedulers = state_dict["n_schedulers"]
            for i in range(n_schedulers):
                self._schedulers[i].lr_scheduler.load_state_dict(
                    state_dict[f"scheduler_{i}"]
                )
            self.loss.load_state_dict(state_dict["loss"])

    # ----------------------------------------
    # Protected methods that may be overloaded
    # ----------------------------------------

    def _create_criterion(self):
        return LossFactory().create_loss(self._config.training.loss)

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def _create_optimizers(self):
        optimizer = OptimizerFactory().create_optimizer(
            self._config.training.optimizer, self.model, self._config
        )

        return [optimizer]

    # overwrite
    def _create_models(self, loader):
        model = ModelFactory().create_model(
            self._config.model.model, self._config, loader, self.device
        )

        return [model]

    def _create_schedulers(self, optimizer):
        scheduler = SchedulerFactory().create_scheduler(
            self._config.training.scheduler, optimizer, self._config
        )
        return [scheduler]

    def _inner_train(self, batch, batch_idx, metrics: MetricsCollection):
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        self.optimizer.zero_grad()
        model_output = self.model(X)

        loss = self.criterion(model_output, Y)
        loss.backward()

        self.optimizer.step()

        metrics.update(model_output, Y)

        return loss.item()

    def _inner_validate(self, batch, batch_idx, metrics: MetricsCollection):
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        model_output = self.model(X)
        loss = self.criterion(model_output, Y)

        metrics.update(model_output, Y)

        return loss.item()

    def _on_epoch_ended(self, epoch, checkpoint, **kwargs):
        lr = self._get_lr(self.optimizer)
        self.logger.report_metric(epoch=epoch, metrics={"learning_rate": lr})

        num_epochs = self._config.training.epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            if "valid_loader" in kwargs and len(self.predictions) > 0:

                self.predictions.compute(
                    self._valuator(kwargs["valid_loader"]))
                self.logger.report_prediction(epoch, self.predictions)

            for item in self.loss(LossType.TRAIN).buffer:
                self.logger.report_metric(
                    epoch=item[0], metrics={f"train_loss": item[1]}
                )
            for item in self.loss(LossType.VALID).buffer:
                self.logger.report_metric(
                    epoch=item[0], metrics={f"valid_loss": item[1]}
                )
            self.loss.flush()

            checkpoint.update(run_id=self.logger.run_id,
                              epoch=epoch, training=self)
            self.logger.log_checkpoint(checkpoint=checkpoint)

    # ----------------------------------------
    # Private methods
    # ----------------------------------------

    def __tqdm_loop(self, function):
        """
        This is a decorator that encapsulates the inner learning procces.
        Iterations over all batches of one epoch.
        This decorator displays a progress bar and computes some times
        """

        def new_function(epoch, loader, description, mean: Mean):
            metrics = MetricsCollection(
                description, self._config)
            metrics.to(self.device)

            timer = Timer()
            timer.start()
            with tqdm(loader, unit="batch", file=sys.stdout) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    tepoch.set_description(f"{description} {epoch}")
                    loss = function(batch, batch_idx, metrics)
                    timer.tick()

                    mean.update(loss)

                    tepoch.set_postfix(loss=mean.value)

            timer.log()
            timer.stop()

            if self.logger.can_report():
                metrics.compute()
                self.logger.report_metrics(epoch, metrics)

        return new_function

    def __batch_loop(self, function):
        """
        This is a decorator that encapsulates the inner learning procces.
        Iterations over all batches of one epoch.
        This decorator is used for learning process in batch mode (less verbose)
        """

        def new_function(epoch, loader, description, mean: Mean):
            size_by_batch = len(loader)
            step = max(size_by_batch //
                       self._config.training.n_steps_by_batch, 1)

            metrics = MetricsCollection(
                description, self._config)
            metrics.to(self.device)

            for batch_idx, batch in enumerate(loader):
                loss = function(batch, batch_idx, metrics)

                mean.update(loss)

                if mean.iter % step == 0:
                    logging.info(
                        f"Epoch {epoch} {description} iter {mean.iter} loss: {mean.value}"
                    )

            if self.logger.can_report():
                metrics.compute()
                self.logger.report_metrics(epoch, metrics)

        return new_function

    def __train(self, epoch, train_loader):
        # Train
        self.model.train()
        return self.train_loop(epoch, train_loader, "Train", self.loss(LossType.TRAIN))

    def __validate(self, epoch, valid_loader):
        self.model.eval()
        with torch.no_grad():
            return self.valid_loop(
                epoch, valid_loader, "Valid", self.loss(LossType.VALID)
            )

    # ----------------------------------------
    # Public methods
    # ----------------------------------------

    def find_lr(self, train_loader):
        num_epochs = self._config.training.epochs
        num_batch = len(train_loader)

        self._models = self._create_models(train_loader)
        self._optimizers = self._create_optimizers()

        lr = self._config.training.learning_rate
        self._config.training.gamma = (
            lr / 1e-8) ** (1 / ((num_batch * num_epochs) - 1))
        self._config.training.step_size = 1
        self.optimizer.param_groups[0]["lr"] = 1e-8
        self._schedulers = [
            SchedulerFactory().create_scheduler("step", self.optimizer, self._config)
        ]

        train_time = Timer()
        with Logger(self._config) as self.logger:
            lrs, losses = [], []
            train_time.start()
            mean_loss = Mean.create("ewm")
            for _ in range(num_epochs):
                with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
                    self.model.train()
                    for batch in tepoch:
                        tepoch.set_description("FindLR")
                        X, Y = batch
                        X = X.to(self.device)
                        Y = Y.to(self.device)
                        self.optimizer.zero_grad()
                        model_output = self.model(X)

                        loss = self.criterion(model_output, Y)
                        loss.backward()
                        self.optimizer.step()

                        lr = self.scheduler.get_last_lr()[0]
                        lv = loss.item()
                        mean_loss.update(lv)
                        lrs.append(lr)
                        losses.append(lv)

                        self.logger.report_metric(
                            mean_loss.iter, {"lr": lr, "loss": lv}
                        )

                        tepoch.set_postfix(loss=lv, lr=lr)

                        train_time.tick()

                        # scheduler update
                        self.scheduler.step()

            self.logger.report_findlr(lrs, losses)
        train_time.log()
        train_time.stop()

    def fit(self, train_loader, valid_loader, run_id=None):
        num_epochs = self._config.training.epochs

        self.loss.reset()

        self._models = self._create_models(train_loader)
        self._optimizers = self._create_optimizers()
        self._schedulers = self._create_schedulers(self.optimizer)

        checkpoint = CheckPoint(run_id)
        checkpoint.init(self)

        with Logger(self._config, run_id=checkpoint.run_id) as self.logger:
            active_run_name = self.logger.active_run_name()
            self._display_device()
            self.logger.set_signature(train_loader)
            self.logger.summary()

            for epoch in range(checkpoint.epoch + 1, num_epochs):

                self.logger.set_epoch(epoch)
                # Train
                self.__train(epoch, train_loader)

                # Test
                self.__validate(epoch, valid_loader)

                # increments scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self._on_epoch_ended(
                    epoch, checkpoint, valid_loader=valid_loader)

            self.logger.save_model(self.model)

        return {
            "run_id": self.logger.run_id,
            "run_name": active_run_name,
            "loss": self.loss(LossType.VALID).value,
        }

    # overwrite
    def predict(self, loader, run_id=None):
        """
        computes predictions for one learned model.
        if run_id is None, reuse a model
        """
        if run_id is not None:
            self._models = self._create_models(loader)
            checkpoint = CheckPoint(run_id)
            checkpoint.init(self, only_model=True)
        else:
            assert self.model is not None, "no model loaded for prediction"

        assert self.predictions is not None, "no predictions defined for this training"
        self.predictions.compute(self, valuator=self._valuator(loader))
