import os
import torch
import mlflow
import tempfile


class CheckPoint:
    def __init__(self, run_id=None) -> None:
        self.params = {"run_id": run_id}
        artifact_uri = self.artifact_uri
        if artifact_uri is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                ck_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=artifact_uri,
                    dst_path=tmpdir
                )
                files = []
                for f in os.listdir(ck_path):
                    ck_name = os.path.join(ck_path, f)
                    if "checkpoint" in f and os.path.isfile(ck_name):
                        files.append(ck_name)
                files.sort(key=lambda x: os.path.getmtime(x))
                ck_name = files[-1]
                self.params = torch.load(
                    ck_name, weights_only=False)
        else:
            self.params["epoch"] = -1

    @property
    def run_id(self):
        return self.params["run_id"]

    @property
    def artifact_uri(self):
        if self.run_id is None:
            return None
        run = mlflow.get_run(self.run_id)
        assert run is not None, f"unable to find run {self.run_id}"
        return os.path.join(run.info.artifact_uri, "checkpoints")

    @property
    def epoch(self):
        return self.params["epoch"]

    def update(self, run_id, epoch, training):
        self.params["run_id"] = run_id
        self.params["epoch"] = epoch
        self.params["training"] = training.state_dict()

    def init(self, training, only_model=False):
        if self.run_id is not None:
            training.load_state_dict(
                self.params["training"], only_model=only_model)
