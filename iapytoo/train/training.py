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
from iapytoo.predictions import Predictions, PredictionPlotter
from iapytoo.metrics.collection import MetricsCollection

from enum import IntEnum


class LossType(IntEnum):
    TRAIN = 0
    VALID = 1


class Training:
    @staticmethod
    def seed(config: Config):
        seed = config.seed
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

    def __init__(
        self,
        config: Config,
        metric_names: list = [],
        prediction_plotter: PredictionPlotter = None,
        y_scaling: Scaling = None,
    ) -> None:
        # first init all random seeds
        seed = config.seed
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._config = config
        self.criterion = self._create_criterion()

        if self.config["tqdm"]:
            self.train_loop = self.__tqdm_loop(self._inner_train)
            self.valid_loop = self.__tqdm_loop(self._inner_validate)
        else:
            self.train_loop = self.__batch_loop(self._inner_train)
            self.valid_loop = self.__batch_loop(self._inner_validate)

        self.logger = None
        self.loss = Loss(n_losses=2)
        self._models = []
        self._optimizers = []
        self._schedulers = []
        self.predictions = None
        self.y_scaling = y_scaling
        self.metric_names = metric_names
        self.prediction_plotter = prediction_plotter

    @property
    def config(self):
        return self._config.__dict__

    @property
    def model(self):
        return self._models[0]

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

    def load_state_dict(self, state_dict):
        n_models = state_dict["n_models"]
        for i in range(n_models):
            self._models[i].load_state_dict(state_dict[f"model_{i}"])
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
        return LossFactory().create_loss(self.config["loss"])

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def _create_optimizers(self):
        optimizer = OptimizerFactory().create_optimizer(
            self.config["optimizer"], self.model, self.config
        )

        return [optimizer]

    def _create_models(self, loader):
        model = ModelFactory().create_model(
            self.config["model"], self.config, loader, self.device
        )

        return [model]

    def _create_schedulers(self, optimizer):
        scheduler = SchedulerFactory().create_scheduler(
            self.config["scheduler"], optimizer, self.config
        )
        return [scheduler]

    def _inner_train(self, batch, batch_idx, metrics: MetricsCollection):
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        self.optimizer.zero_grad()
        Y_hat = self.model(X)

        loss = self.criterion(Y_hat, Y)
        loss.backward()

        self.optimizer.step()

        metrics.update(Y_hat, Y)

        return loss.item()

    def _inner_validate(self, batch, batch_idx, metrics: MetricsCollection):
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        Y_hat = self.model(X)
        loss = self.criterion(Y_hat, Y)

        metrics.update(Y_hat, Y)

        return loss.item()

    def _on_epoch_ended(self, epoch, checkpoint):
        lr = self._get_lr(self.optimizer)
        self.logger.report_metric(epoch=epoch, metrics={"learning_rate": lr})

        if epoch % 10 == 0:
            self.predictions.compute(self)
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

            checkpoint.update(run_id=self.logger.run_id, epoch=epoch, training=self)
            self.logger.log_checkpoint(checkpoint=checkpoint)

    # ----------------------------------------
    # Private methods
    # ----------------------------------------

    def _display_device(self):
        use_cuda = torch.cuda.is_available()
        if self.config["cuda"] and use_cuda:
            msg = "\n__CUDA\n"
            msg += f"__CUDNN VERSION: {torch.backends.cudnn.version()}\n"
            msg += f"__Number CUDA Devices: {torch.cuda.device_count()}\n"
            msg += f"__CUDA Device Name: {torch.cuda.get_device_name(0)}\n"
            msg += f"__CUDA Device Total Memory [GB]: {torch.cuda.get_device_properties(0).total_memory / 1e9}\n"
            msg += "-----------\n"
            logging.info(msg)
        else:
            logging.info("__CPU")

    def __tqdm_loop(self, function):
        """
        This is a decorator that encapsulates the inner learning procces.
        Iterations over all batches of one epoch.
        This decorator displays a progress bar and computes some times
        """

        def new_function(epoch, loader, description, mean: Mean):
            metrics = MetricsCollection(
                description, self.metric_names, self.config, loader
            )
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
            step = max(size_by_batch // self.config["n_steps_by_batch"], 1)

            metrics = MetricsCollection(
                description, self.metric_names, self.config, loader
            )
            metrics.to(self.device)

            for batch_idx, batch in enumerate(loader):
                loss = function(batch, batch_idx, metrics)

                mean.update(loss)

                if mean.iter % step == 0:
                    logging.info(
                        f"Epoch {epoch} {description} iter {mean.iter} loss: {mean.value}"
                    )

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
        num_epochs = self.config["epochs"]
        num_batch = len(train_loader)

        self._models = self._create_models(train_loader)
        self._optimizers = self._create_optimizers()

        lr = self.config["learning_rate"]
        self.config["gamma"] = (lr / 1e-8) ** (1 / ((num_batch * num_epochs) - 1))
        self.optimizer.param_groups[0]["lr"] = 1e-8
        self._schedulers = [
            SchedulerFactory().create_scheduler("step", self.optimizer, self.config)
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
                        Y_hat = self.model(X)

                        loss = self.criterion(Y_hat, Y)
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
        num_epochs = self.config["epochs"]

        self.loss.reset()

        self._models = self._create_models(train_loader)
        self._optimizers = self._create_optimizers()
        self._schedulers = self._create_schedulers(self.optimizer)

        self.predictions = Predictions(
            valid_loader, prediction_plotter=self.prediction_plotter
        )

        checkpoint = CheckPoint(run_id)
        checkpoint.init(self)

        with Logger(self._config, run_id=checkpoint.run_id) as self.logger:
            active_run_name = self.logger.active_run_name()
            self._display_device()
            self.logger.set_signature(train_loader)
            self.logger.summary()

            for epoch in range(checkpoint.epoch + 1, num_epochs):
                # Train
                self.__train(epoch, train_loader)

                # Test
                self.__validate(epoch, valid_loader)

                # increments scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self._on_epoch_ended(epoch, checkpoint)

            self.logger.save_model(self.model)

        return {
            "run_id": self.logger.run_id,
            "run_name": active_run_name,
            "loss": self.loss(LossType.VALID).value,
        }

    def predict(self, loader, run_id=None):
        """
        computes predictions for one learned model.
        if run_id is None, reuse a model
        """
        if run_id is not None:
            self._models = self._create_models(loader)
            checkpoint = CheckPoint(run_id)
            checkpoint.init_model(self.model)
        else:
            assert self.model is not None, "no model loaded for prediction"

        self.predictions = Predictions(loader)
        self.predictions.compute(self)
