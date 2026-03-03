import asyncio
from mlserver import MLServer
from mlserver.settings import Settings, ModelSettings

from pathlib import Path

root_path = Path(__file__).parent


async def main():

    settings = Settings.model_validate(
        {
            "debug": True,
            # optionnel :
            "parallel_workers": 0,
        }
    )

    models_settings = ModelSettings.model_validate(
        {
            "name": "wgan",
            "implementation": "mlserver_mlflow.MLflowRuntime",
            "parameters": {
                "uri": str(root_path / "models" / "m-f095bb8a45ad4320a05b65a4b6d2461a" / "artifacts")
            }
        }
    )

    server = MLServer(settings=settings)

    # # 👉 MET TON BREAKPOINT ICI
    # import pdb
    # pdb.set_trace()

    await server.start(models_settings=[models_settings])

if __name__ == "__main__":
    asyncio.run(main())
