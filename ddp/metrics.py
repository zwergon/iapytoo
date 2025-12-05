import os
import torch
from pathlib import Path

import torch.distributed as dist

from iapytoo.metrics.collection import MetricsCollection
from iapytoo.utils.config import Config, ConfigFactory
from iapytoo.train.training import Training


def distributed():
    return 'RANK' in os.environ


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if distributed():
        rank = int(os.environ['RANK'])
        Training.setup_ddp(rank)
    else:
        rank = 0

    config: Config = ConfigFactory.from_yaml(
        Path(__file__).parent / "config.yml")

    collection = MetricsCollection("ddp", config.metrics.names, config)
    collection.to(device)

    actual_list = [3.0, -0.5, 2.0, 7.0]
    predicted_list = [2.5, 0.0, 2.0, 8.0]

    if dist.is_initialized():
        if rank == 0:
            actual = torch.tensor(actual_list[:2]).to(device)
            predicted = torch.tensor(predicted_list[:2]).to(device)
        elif rank == 1:
            actual = torch.tensor(actual_list[2:]).to(device)
            predicted = torch.tensor(predicted_list[2:]).to(device)
    else:
        actual = torch.tensor(actual_list).to(device)
        predicted = torch.tensor(predicted_list).to(device)

    collection.update(predicted, actual)

    result = collection.compute()

    print(f"[rank {rank}] {result}")
