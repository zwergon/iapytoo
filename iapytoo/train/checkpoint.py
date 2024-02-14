import os
import torch
import mlflow
import tempfile


class CheckPoint:
    def __init__(self, run_id=None) -> None:
        self.params = {"run_id": run_id}
        if run_id is not None:
            run = mlflow.get_run(self.run_id)
            assert run is not None, f"unable to find run {self.run_id}"
            with tempfile.TemporaryDirectory() as tmpdir:
                local_artifact = mlflow.artifacts.download_artifacts(
                    run.info.artifact_uri, dst_path=tmpdir
                )
                ck_path = os.path.join(local_artifact, "checkpoints", "checkpoint.pt")
                self.params = torch.load(ck_path)
        else:
            self.params["epoch"] = -1

    @property
    def run_id(self):
        return self.params["run_id"]

    @property
    def epoch(self):
        return self.params["epoch"]

    def init_model(self, model):
        if self.run_id is not None:
            model.load_state_dict(self.params["model"])

    def init_optimizer(self, optimizer):
        if self.run_id is not None:
            optimizer.load_state_dict(self.params["optimizer"])

    def init_scheduler(self, scheduler):
        if self.run_id is not None:
            scheduler.load_state_dict(self.params["scheduler"])

    def init_loss(self, train_loss, valid_loss):
        if self.run_id is not None:
            train_loss.load_state_dict(self.params["train_loss"])
            valid_loss.load_state_dict(self.params["valid_loss"])

    def update(self, **kwargs):
        self.params.update(**kwargs)
