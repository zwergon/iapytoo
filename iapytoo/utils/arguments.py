import argparse

from iapytoo.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to parquet file")
    parser.add_argument(
        "-e",
        "--epochs",
        help="number of epochs in learning process",
        type=int,
        default=101,
    )
    parser.add_argument(
        "-b", "--batch_size", help="batch size in learning process", type=int, default=8
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="learning rate for optimizer",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "-wd", "--weight_decay", help="regularization", type=float, default=1e-7
    )

    parser.add_argument(
        "-c",
        "--config",
        help="training config file",
        type=str,
        default=Config.default_path(),
    )
    return parser.parse_args()