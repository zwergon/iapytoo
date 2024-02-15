import sys
import random
import numpy
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm


from iapytoo.utils.config import Config
from iapytoo.utils.timer import Timer
from iapytoo.utils.iterative_mean import Mean
from iapytoo.train.logger import Logger
from iapytoo.train.checkpoint import CheckPoint
from iapytoo.train.predictions import Predictions, PredictionPlotter
from iapytoo.train.metrics_collection import MetricsCollection
from iapytoo.train.models import ModelFactory


class Training:

    @staticmethod
    def seed(config: Config):
        seed = config.seed
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

    def __init__(self, 
        config: Config, 
        metric_creators: list = None,
        prediction_plotter: PredictionPlotter = None ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._config = config
        self.criterion = self._create_criterion()

        if self.config["tqdm"]:
            self.train_loop = self.__tqdm_loop(self._inner_train)
            self.valid_loop = self.__tqdm_loop(self._inner_validate)
        else:
            self.train_loop = self.__batch_loop(self._inner_train)
            self.valid_loop = self.__batch_loop(self._inner_validate)

        self.model = None
        self.logger = None
        self.optimizer = None
        self.scheduler = None
        self.predictions = None
        self.metric_creators = metric_creators
        self.prediction_plotter = prediction_plotter
 
    @property
    def config(self):
        return self._config.__dict__
    
    # ----------------------------------------
    # Protected methods that may be overloaded
    # ----------------------------------------

    
    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]
    
    def _create_optimizer(self):
        
        model = self.model
        if self.config["optimizer"] == "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            raise Exception("Unknown optimizer")

        self.optimizer = optimizer
        return optimizer
    
    def _create_criterion(self):
        return nn.MSELoss()
    

    def _create_model(self, loader):

        model = ModelFactory().create_model(
            self.config, 
            loader, 
            self.device
            )
        
        return model

    def _create_scheduler(self, optimizer):

        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        def lr_lambda(epoch):
            # LR to be 0.1 * (1/1+0.01*epoch)
            return 0.995**epoch

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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

    def _inner_validate(self, batch, batch_idx, metrics : MetricsCollection):
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        Y_hat = self.model(X)
        loss = self.criterion(Y_hat, Y)

        metrics.update(Y_hat, Y)

        return loss.item()
    
    def _on_epoch_ended(self, epoch, checkpoint, train_loss, valid_loss):
        
        if epoch % 10 == 0:
            self.predictions.compute(self)
            self.logger.report_prediction(epoch, self.predictions)

            checkpoint.update(
                        run_id=self.logger.run_id,
                        epoch=epoch,
                        model=self.model.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        scheduler=self.scheduler.state_dict(),
                        train_loss=train_loss.state_dict(),
                        valid_loss=valid_loss.state_dict()
                    )
            self.logger.log_checkpoint(checkpoint=checkpoint)
    
    # ----------------------------------------
    # Private methods
    # ----------------------------------------

    def __display_device(self):

        use_cuda = torch.cuda.is_available()
        if self.config['cuda'] and use_cuda:
                print('__CUDNN VERSION:', torch.backends.cudnn.version())
                print('__Number CUDA Devices:', torch.cuda.device_count())
                print('__CUDA Device Name:',torch.cuda.get_device_name(0))
                print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
        else:
                print('__CPU')

    def __tqdm_loop(self, function):
        """
        This is a decorator that encapsulates the inner learning procces.
        Iterations over all batches of one epoch.
        This decorator displays a progress bar and computes some times
        """

        def new_function(epoch, loader, description, mean: Mean):
            
            size_by_batch = len(loader)
            step = max(size_by_batch // self.config["n_steps_by_batch"], 1)

            metrics = MetricsCollection(description, self.metric_creators)
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
                    if mean.iter % step == 0:
                        self.logger.report_metric(
                            epoch=mean.iter, metrics={f"{description}_loss": mean.value}
                        )
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

            metrics = MetricsCollection(description, self.metric_creators)
            metrics.to(self.device)
            
            for batch_idx, batch in enumerate(loader):
                loss = function(batch, batch_idx, self.metrics)

                mean.update(loss)

                if mean.iter % step == 0:
                    print(f"{description} iter {mean.iter} loss: {mean.value}")
                    self.logger.report_metric(
                        epoch=mean.iter, metrics={f"{description}_loss": mean.value}
                    )
            
            metrics.compute()
            self.logger.report_metrics(epoch, metrics)
            
        return new_function
    
    
    def __train(self, epoch, train_loader, train_loss):
        # Train
        self.model.train()
        return self.train_loop(epoch, train_loader, "Train", train_loss)

    def __validate(self, epoch, valid_loader, valid_loss):
        self.model.eval()
        with torch.no_grad():
            return self.valid_loop(epoch, valid_loader, "Valid", valid_loss)
        
    # ----------------------------------------
    # Public methods
    # ----------------------------------------

    
    def find_lr(self, train_loader):
        num_epochs = self.config["epochs"]
        num_batch = len(train_loader)

        self.model = self._create_model(train_loader)
        self.optimizer = self._create_optimizer()

        lr = self.config["learning_rate"]
        mult = (lr / 1e-8) ** (1 / ((num_batch * num_epochs) - 1))
        self.optimizer.param_groups[0]["lr"] = 1e-8
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=mult
        )

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

        train_loss = Mean.create("ewm")
        valid_loss = Mean.create("ewm")
        
        self.model = self._create_model(train_loader)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(self.optimizer)

        self.predictions = Predictions(valid_loader, prediction_plotter=self.prediction_plotter)

        checkpoint = CheckPoint(run_id)
        checkpoint.init_model(self.model)
        checkpoint.init_optimizer(self.optimizer)
        checkpoint.init_scheduler(self.scheduler)
        checkpoint.init_loss(train_loss, valid_loss)


        with Logger(self._config, run_id=None) as self.logger:
            self.__display_device()
            self.logger.set_signature(train_loader)
            self.logger.summary()

            for epoch in range(num_epochs):

                # Train   
                self.__train(epoch, train_loader, train_loss)
            
                # Test
                self.__validate(epoch, valid_loader, valid_loss)

                # increments scheduler
                self.scheduler.step()
                
                self._on_epoch_ended(epoch, checkpoint, train_loss, valid_loss)

            self.logger.save_model(self.model)

    def predict(self, loader, run_id):

        self.model = self._create_model(loader)
        self.predictions = Predictions(loader)

        checkpoint = CheckPoint(run_id)
        checkpoint.init_model(self.model)

        self.predictions.compute(self)

    
