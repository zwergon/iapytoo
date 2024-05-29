import argparse

from iapytoo.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="root path to the dataset")
    parser.add_argument(
        "-e",
        "--epochs",
        help="number of epochs in learning process",
        type=int,
        default=11,
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

    parser.add_argument("-r", "--run_id", help="which run_id to continue", default=None)
    parser.add_argument(
        "-tu",
        "--tracking_uri",
        help="where to find run_id",
        type=str,
        default=None,
    )
    return parser.parse_args()
