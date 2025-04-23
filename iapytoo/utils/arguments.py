import argparse

from iapytoo.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run with either a config file or individual parameters."
    )

    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument(
        "--yaml", type=str, help="Path to the YAML configuration file."
    )

    # Creating a subgroup to avoid required=True inside the mutually exclusive group
    param_group = exclusive_group.add_argument_group("Individual parameters")
    param_group.add_argument("--run-id", type=str, help="Run identifier.")
    param_group.add_argument("--epochs", type=int, help="Number of epochs.")
    param_group.add_argument(
        "--tracking-uri", type=str, help="Tracking URI (optional)."
    )

    args = parser.parse_args()

    # Manual validation: If --yaml is not provided, both --run-id and --epochs are required
    if args.yaml is None and (args.run_id is None or args.epochs is None):
        parser.error(
            "If --yaml is not provided, both --run-id and --epochs are required."
        )

    return args
