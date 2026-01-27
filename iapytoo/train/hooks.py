import logging
import sys
from tqdm import tqdm

from typing import Protocol
from iapytoo.metrics.collection import MetricsCollection
from iapytoo.train.loss import Loss


class TrainHook(Protocol):
    def on_epoch_start(self, epoch: int, description: str): ...
    def on_batch_start(self, batch_idx: int, total: int): ...
    def on_batch_end(self, batch_idx: int, total: int, loss: float): ...
    def on_epoch_end(self, epoch: int, metrics: MetricsCollection): ...


class TqdmHook:
    def __init__(self):
        self.pbar = None

    def on_epoch_start(self, epoch, description):
        self.pbar = tqdm(unit="batch", file=sys.stdout)
        self.pbar.set_description(f"{description} {epoch}")

    def on_batch_start(self, batch_idx, total):
        if self.pbar.total is None:
            self.pbar.total = total

    def on_batch_end(self, batch_idx, total, losses: dict):
        self.pbar.update(1)
        self.pbar.set_postfix(**losses)

    def on_epoch_end(self, epoch, metrics):
        self.pbar.close()


class LoggingHook:
    def __init__(self, every_n=50):
        self.every_n = every_n

    def on_epoch_start(self, epoch, description):
        logging.info(f"Epoch {epoch} {description}")

    def on_batch_start(self, batch_idx, total):
        pass

    def on_batch_end(self, batch_idx, total, losses: dict):
        msg = f"Step {batch_idx+1}/{total}"
        for k, v in losses.items():
            msg += f" - {k}: {v:.6f}"

        if batch_idx % self.every_n == 0:
            logging.info(msg)

    def on_epoch_end(self, epoch, metrics):
        logging.info("Epoch finished")
