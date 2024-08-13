import os
import mlflow


class Context:

    filename = "context.txt"

    def __init__(self, run_id=None) -> None:
        self.__dict__["run_id"] = run_id
        self.last_epoch = -1
        if run_id is not None:
            run = mlflow.get_run(self.run_id)
            assert run is not None, f"unable to find run {self.run_id}"

            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=Context.filename
            )
            with open(local_path, "r") as f:
                for line in f:
                    key, value = line.strip().split(": ", 1)
                    self.__dict__[key] = value

    def write(self, dirname):
        filepath = os.path.join(dirname, Context.filename)
        with open(filepath, "w") as f:
            for key, value in self.__dict__.items():
                f.write(f"{key}: {value}\n")
        return filepath

    def can_report(self, epoch) -> bool:
        return epoch > int(self.last_epoch)
