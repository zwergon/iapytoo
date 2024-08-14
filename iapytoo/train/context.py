import os
import mlflow


class Context:

    filename = "context.txt"

    def __init__(self, run_id=None) -> None:
        self.run_id = run_id
        self.last_epoch = -1
        self.epoch = -1
        if run_id is not None:
            self._load_from_artifact(run_id)

    def _load_from_artifact(self, run_id):
        run = mlflow.get_run(self.run_id)
        assert run is not None, f"unable to find run {self.run_id}"

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=Context.filename
        )
        with open(local_path, "r") as f:
            for line in f:
                key, value = line.strip().split(": ", 1)
                if key == "epoch":
                    self.last_epoch = int(value)

    def save(self, dirname):
        filepath = os.path.join(dirname, Context.filename)
        context_dict = {"epoch": self.epoch}
        with open(filepath, "w") as f:
            for key, value in context_dict.items():
                f.write(f"{key}: {value}\n")
        return filepath
