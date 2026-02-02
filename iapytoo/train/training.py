from iapytoo.train.mlflow_model import save_mlflow_model
from iapytoo.metrics.metric import Metrics
from iapytoo.train.inference import Inference
from iapytoo.train.checkpoint import CheckPoint
from iapytoo.train.logger import Logger
from iapytoo.train.factories import Factory
from iapytoo.train.hooks import TrainHook, TqdmHook, LoggingHook
from iapytoo.train.loss import Loss
from iapytoo.utils.iterative_mean import Mean
from iapytoo.utils.timer import Timer
from iapytoo.utils.config import Config, TrainingConfig
from iapytoo.train.scheduler import Scheduler
from torch.utils.data import DataLoader

import os
import sys
import random
import numpy
import torch
import logging
from tqdm import tqdm
from enum import Enum

from typing import List


class LossType(str, Enum):
    TRAIN = "train_loss"
    VALID = "valid_loss"


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

    @staticmethod
    def setup_ddp(rank):
        import torch.distributed as dist
        world_size = int(os.environ['WORLD_SIZE'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']

        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        # Choisir NCCL si GPU disponible, sinon Gloo
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(
            backend=backend, rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Process initialized using {backend} backend")

    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

        # first init all random seeds
        self.seed(config)

        self.criterion = self._create_criterion()

        train_func = torch.compile(
            self._inner_train) if config.training.compile else self._inner_train

        if self._config.training.tqdm:
            self.hooks: list[TrainHook] = [TqdmHook()]
        else:
            self.hooks: list[TrainHook] = [LoggingHook()]

        self.train_loop = self.__hooked_loop(train_func)
        self.valid_loop = self.__hooked_loop(self._inner_validate)

        self.loss = Loss(LossType, config.plotting_mean)
        self._optimizers = []
        self._schedulers = []
        self._metrics = {}

    @property
    def scheduler(self) -> Scheduler:
        if len(self._schedulers) > 0:
            return self._schedulers[0]

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
        training_conf: TrainingConfig = self._config.training
        criterion = Factory().create_loss(training_conf.loss, self._config)
        criterion.to(self.device)
        return criterion

    def _get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def _create_optimizers(self):
        optimizer = Factory().create_optimizer(
            self._config.training.optimizer, self.model, self._config
        )

        return [optimizer]

    # overwrite

    def _create_models(self):
        assert self.mlflow_model_provider is not None, "all model definition in provider"
        model = self.mlflow_model_provider.model
        model.to(self.device)

        return [model]

    def _create_schedulers(self, optimizer):
        scheduler = Factory().create_scheduler(
            self._config.training.scheduler, optimizer, self._config
        )
        return [scheduler]

    def _create_metrics(self):
        return {
            "Train": Factory().create_metrics(
                "Train",
                self._config.metrics.names,
                self._config,
                predictor=self.predictor,
                device=self.device
            ),
            "Valid": Factory().create_metrics(
                "Valid",
                self._config.metrics.names,
                self._config,
                predictor=self.predictor,
                device=self.device
            )
        }

    def _inner_train(self, batch, batch_idx):
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        self.optimizer.zero_grad()
        model_output = self.model(X)

        loss = self.criterion(model_output, Y)
        loss.backward()

        self.optimizer.step()

        f_loss = loss.item()

        if self.scheduler is not None:
            self.scheduler.update(f_loss)

        try:
            metrics: Metrics = self._metrics["Train"]
            metrics.update(model_output, Y)
        except KeyError:
            pass

        self.loss(LossType.TRAIN).update(f_loss)
        return {LossType.TRAIN: f_loss}

    def _inner_validate(self, batch, batch_idx):
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        model_output = self.model(X)
        loss = self.criterion(model_output, Y)

        try:
            metrics: Metrics = self._metrics["Valid"]
            metrics.update(model_output, Y)
        except KeyError:
            pass

        self.loss(LossType.VALID).update(loss.item())
        return {LossType.VALID: loss.item()}

    def _train(self, epoch, train_loader):
        # Train
        self.model.train()
        return self.train_loop(epoch, train_loader, "Train")

    def _validate(self, epoch, valid_loader):
        self.model.eval()
        with torch.no_grad():
            return self.valid_loop(epoch, valid_loader, "Valid")

    def _on_epoch_ended(self, epoch, checkpoint, checkpoint_epoch=None, report_per_epoch=None, **kwargs):
        lr = self._get_lr(self.optimizer)
        self.logger.report_metric(epoch=epoch, metrics={"learning_rate": lr})

        num_epochs = self._config.training.epochs

        if report_per_epoch:
            self._report_metrics(epoch, **kwargs)

        if checkpoint_epoch is not None and \
                (epoch % checkpoint_epoch == 0 or epoch == num_epochs - 1):
            checkpoint.update(run_id=self.logger.run_id,
                              epoch=epoch, training=self)
            self.logger.log_checkpoint(checkpoint=checkpoint)

            if not report_per_epoch:
                self._report_metrics(epoch, **kwargs)
        if checkpoint_epoch is None and not report_per_epoch and epoch == num_epochs - 1:
            self._report_metrics(epoch, **kwargs)

    def _report_metrics(self, epoch, **kwargs):
        if "loader" in kwargs and len(self.predictions) > 0:
            self.predictions.compute(loader=kwargs["loader"])
            self.logger.report_prediction(epoch, self.predictions)

        for lt in self.loss.enum_cls:
            for item in self.loss(lt).get_loss():
                key: str = str(lt)
                self.logger.report_metric(epoch=item[0], metrics={
                    key: item[1]})
        self.loss.flush()

    # ----------------------------------------
    # Private methods
    # ----------------------------------------

    def __hooked_loop(self, function):
        def new_function(epoch, loader, description):
            hooks = self.hooks or []

            try:
                metrics: Metrics = self._metrics[description]
                metrics.reset()
            except KeyError:
                logging.warning(f"No metrics defined for {description} phase")
                metrics = None

            timer = Timer()
            timer.start()

            for h in hooks:
                h.on_epoch_start(epoch, description)

            total = len(loader)

            for batch_idx, batch in enumerate(loader):
                for h in hooks:
                    h.on_batch_start(batch_idx, total)

                losses = function(batch, batch_idx)
                timer.tick()

                for h in hooks:
                    h.on_batch_end(batch_idx, total, losses)

            if self.logger.can_report() and metrics is not None:
                metrics.compute()
                self.logger.report_metrics(epoch, metrics)

            for h in hooks:
                h.on_epoch_end(epoch, metrics)

            timer.log()
            timer.stop()

        return new_function

    # ----------------------------------------
    # Public methods
    # ----------------------------------------

    def find_lr(self, train_loader):
        num_epochs = self._config.training.epochs
        num_batch = len(train_loader)

        self._models = self._create_models()
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
            mean_loss = Mean.create(self._config.plotting_mean)
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
        self, train_loader: DataLoader, valid_loader: DataLoader = None, run_id=None
    ):
        num_epochs = self._config.training.epochs

        self.loss.reset()
        self._models = self._create_models()
        self._metrics = self._create_metrics()
        self._optimizers = self._create_optimizers()
        self._schedulers = self._create_schedulers(self.optimizer)

        checkpoint = CheckPoint(run_id)
        checkpoint.init(self)
        checkpoint_epoch = self._config.checkpoint_epoch
        report_per_epoch = True if self.loss.plotting_mean in [
            "raw_loss", "epoch"] else False
        save_model = self._config.save_model

        with Logger(self._config, run_id=checkpoint.run_id) as self.logger:
            active_run_name = self.logger.active_run_name()
            self._display_device()
            self.logger.summary()

            for epoch in range(checkpoint.epoch + 1, num_epochs):

                self.logger.set_epoch(epoch)

                # Train
                self._train(epoch, train_loader)

                # Validate
                if valid_loader is not None:
                    self._validate(epoch, valid_loader)

                # increments scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                self._on_epoch_ended(
                    epoch, checkpoint,
                    checkpoint_epoch=checkpoint_epoch,
                    report_per_epoch=report_per_epoch,
                    loader=valid_loader
                )

            if save_model:
                save_mlflow_model(
                    self._config,
                    provider=self.mlflow_model_provider,
                    epoch=num_epochs
                )

        status = {
            "run_id": self.logger.run_id,
            "run_name": active_run_name
        }
        status.update(self.loss.to_dict())
        logging.info(status)
        return status

    # overwrite
    def predict(self, loader, run_id=None):
        """
        computes predictions for one learned model.
        if run_id is None, reuse a model
        """
        if run_id is not None:
            self._models = self._create_models()
            checkpoint = CheckPoint(run_id)
            checkpoint.init(self, only_model=True)
        else:
            assert self.model is not None, "no model loaded for prediction"

        assert self.predictions is not None, "no predictions defined for this training"
        self.predictions.compute(loader=loader)
