import sys
import random
import numpy
import torch
import logging
from tqdm import tqdm
from enum import IntEnum

from torch.utils.data import DataLoader

from iapytoo.utils.config import Config
from iapytoo.utils.timer import Timer
from iapytoo.utils.iterative_mean import Mean
from iapytoo.train.loss import Loss
from iapytoo.train.factories import Factory
from iapytoo.train.logger import Logger
from iapytoo.train.checkpoint import CheckPoint
from iapytoo.train.inference import Inference
from iapytoo.metrics.collection import MetricsCollection


class LossType(IntEnum):
    TRAIN = 0
    VALID = 1


class Training(Inference):
    @staticmethod
    def seed(config: Config):
        seed = config.seed
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

        # first init all random seeds
        self.seed(config)

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
                    state_dict[f"optimizer_{i}"])
            n_schedulers = state_dict["n_schedulers"]
            for i in range(n_schedulers):
                self._schedulers[i].lr_scheduler.load_state_dict(
                    state_dict[f"scheduler_{i}"])
            self.loss.load_state_dict(state_dict["loss"])

    # ----------------------------------------
    # Protected methods that may be overloaded
    # ----------------------------------------

    def _create_criterion(self):
        return Factory().create_loss(self._config.training.loss)

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def _create_optimizers(self):
        optimizer = Factory().create_optimizer(
            self._config.training.optimizer, self.model, self._config
        )

        return [optimizer]

    # overwrite
    def _create_models(self, loader):
        model = Factory().create_model(self._config.model.model,
                                       self._config, loader, self.device)

        return [model]

    def _create_schedulers(self, optimizer):
        scheduler = Factory().create_scheduler(
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

                self.predictions.compute(loader=kwargs["valid_loader"])
                self.logger.report_prediction(epoch, self.predictions)

            for item in self.loss(LossType.TRAIN).buffer:
                self.logger.report_metric(epoch=item[0], metrics={
                                          "train_loss": item[1]})
            for item in self.loss(LossType.VALID).buffer:
                self.logger.report_metric(epoch=item[0], metrics={
                                          "valid_loss": item[1]})
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
                description, self._config.metrics.names, self._config)
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
                description, self._config.metrics.names, self._config)
            metrics.to(self.device)

            for batch_idx, batch in enumerate(loader):
                loss = function(batch, batch_idx, metrics)

                mean.update(loss)

                if mean.iter % step == 0:
                    logging.info(
                        f"Epoch {epoch} {description} iter {mean.iter} loss: {mean.value}")

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
            return self.valid_loop(epoch, valid_loader, "Valid", self.loss(LossType.VALID))

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
        self._schedulers = [Factory().create_scheduler(
            "step", self.optimizer, self._config)]

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
                            mean_loss.iter, {"lr": lr, "loss": lv})

                        tepoch.set_postfix(loss=lv, lr=lr)

                        train_time.tick()

                        # scheduler update
                        self.scheduler.step()

            self.logger.report_findlr(lrs, losses)
        train_time.log()
        train_time.stop()

    def fit(
        self, train_loader: DataLoader, valid_loader: DataLoader, run_id=None
    ):
        num_epochs = self._config.training.epochs

        self.loss.reset()

        self._models = self._create_models(loader=train_loader)
        self._optimizers = self._create_optimizers()
        self._schedulers = self._create_schedulers(self.optimizer)
        self._init_mlflow_model()

        checkpoint = CheckPoint(run_id)
        checkpoint.init(self)

        with Logger(self._config, run_id=checkpoint.run_id) as self.logger:
            active_run_name = self.logger.active_run_name()
            self._display_device()
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

            self.logger.save_model(self.mlflow_model)

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
        self.predictions.compute(loader=loader)
