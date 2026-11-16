import mlflow
import numpy as np

from iapytoo.train.inference import get_model_uri
from iapytoo.mlflow.model import MlInput
from mlflow.tracking import MlflowClient


if __name__ == "__main__":
    import argparse
    import mlflow.pyfunc as mp
    import matplotlib.pyplot as plt

    from iapytoo.train.factories import Factory
    from examples.fm.backbone import VelocityUNet1D

    parser = argparse.ArgumentParser(description="Generate sines from a flow matching run")
    parser.add_argument("--run-id", required=True, help="MLflow run_id")
    parser.add_argument("--output", type=str, default="fm.jpg",
                        help="output filename for a few generated signals")
    args = parser.parse_args()

    # The provider rebuilds FlowMatchingModel, whose backbone must be registered.
    Factory().register_backbone("velocity_unet1d", VelocityUNet1D)

    run = MlflowClient().get_run(args.run_id)
    n_channels = int(run.data.params['model.n_channels'])
    signal_length = int(run.data.params['model.signal_length'])

    loaded_model: mp.PyFuncModel = mlflow.pyfunc.load_model(get_model_uri(args.run_id))

    # Inputs are the starting points x(0) ~ N(0, I) of the ODE.
    noise = np.random.randn(4, n_channels, signal_length).astype(np.float32)
    predicted = loaded_model.predict([MlInput.from_array(x) for x in noise])

    plt.figure()
    for signal in predicted:
        plt.plot(signal[0, :])
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
