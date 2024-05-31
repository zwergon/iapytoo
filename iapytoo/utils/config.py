import json
import os
import sys
import logging
import mlflow
import tempfile


os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"


class Config:
    @staticmethod
    def create_from_args(args):
        config = Config(args.config)
        config.__dict__.update(args.__dict__)
        return config

    @staticmethod
    def _indices(vars):
        v_list = vars[1:-1].split(",")
        return [int(v) for v in v_list]

    @staticmethod
    def _bool(var):
        if var == "True":
            return True
        else:
            return False

    @staticmethod
    def create_from_run_id(run_id, tracking_uri=None):
        cf = {}

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        run = mlflow.get_run(run_id)

        for key, value in run.data.params.items():
            try:
                cf[key] = {
                    "project": str,
                    "run": str,
                    "sensors": str,
                    "tracking_uri": str,
                    "tqdm": Config._bool,
                    "normalization": Config._bool,
                    "n_steps_by_batch": int,
                    "ratio_train_test": float,
                    "num_workers": int,
                    "model": str,
                    "scheduler": str,
                    "gamma": float,
                    "lambda_gp": float,
                    "hidden_size": int,
                    "num_layers": int,
                    "kernel_size": int,
                    "groups": int,
                    "dropout": float,
                    "loss": str,
                    "optimizer": str,
                    "cuda": Config._bool,
                    "seed": int,
                    "batch_size": int,
                    "learning_rate": float,
                    "weight_decay": float,
                    "indices": Config._indices,
                }[key](value)

            except KeyError:
                logging.warning(f"key {key} not converted")

        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        json.dump(cf, temp_file, indent=4)
        temp_file.close()
        config = Config(temp_file.name)
        os.unlink(temp_file.name)

        return config

    def __repr__(self) -> str:
        str = "\nConfig:\n"
        for k, v in self.__dict__.items():
            str += f".{k}: {v}\n"
        str += "---------\n"
        return str

    @staticmethod
    def default_path():
        return os.path.join(os.path.dirname(__file__), "cf", "config.json")

    @staticmethod
    def default_config():
        return Config(Config.default_path())

    @staticmethod
    def test_config():
        return Config(os.path.join(os.path.dirname(__file__), "cf", "config_test.json"))

    def __init__(self, config_name) -> None:
        with open(config_name, "r") as file:
            self.__dict__ = json.load(file)

        # initialize root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if len(logger.handlers) > 0:
            logger.handlers[0].setFormatter(
                logging.Formatter("[%(levelname)s]  %(message)s")
            )
        else:
            sh = logging.StreamHandler(stream=sys.stdout)
            sh.setFormatter(logging.Formatter("[%(levelname)s]  %(message)s"))
            logger.addHandler(sh)
