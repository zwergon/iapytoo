import subprocess
import re
import yaml
import os
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Any, List
from enum import IntEnum


class StepStatus(IntEnum):
    FAIL = -1
    OK = 0
    TEST = 2
    SKIP = 3


PYTHON = sys.executable
root_dir = Path(__file__).parent

tmp_dir = root_dir / 'tmp'


class StepEnum(IntEnum):
    MNIST = 0
    MNIST_AGAIN = 1
    MNIST_INFER = 2
    MNIST_MLFLOW = 3
    WGAN = 4
    WGAN_AGAIN = 5
    WGAN_MLFLOW = 6
    DDPM_TRAIN = 7
    DDPM_MLFLOW = 8
    MNIST_FIND_LR = 9
    LAST = 10


# mnist, mnist_again, mnist_infer, mnist_mlflow_infer, wgan_train, wgan_mlflow
actions = [True] * StepEnum.LAST
# actions[StepEnum.MNIST] = False
# actions[StepEnum.MNIST_AGAIN] = False
# actions[StepEnum.MNIST_INFER] = False
# actions[StepEnum.MNIST_MLFLOW] = False
# actions[StepEnum.WGAN] = False
actions[StepEnum.WGAN_AGAIN] = False
# actions[StepEnum.WGAN_MLFLOW] = False
# actions[StepEnum.DDPM_TRAIN] = False
# actions[StepEnum.DDPM_MLFLOW] = False
# actions[StepEnum.MNIST_FIND_LR] = False


@dataclass
class Step:
    name: str
    func: Callable[..., Any]
    script: Path | None = None
    config: Path | None = None
    log_file: Path | None = None
    needs_run_id: bool = False
    returns_run_id: bool = False
    extra_args: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.SKIP

    def _extract_run_id(self):
        with open(self.log_file, "r") as f:
            for line in f:
                match = re.search(r"Run_id:\s*([a-f0-9]+)", line)
                if match:
                    run_id = match.group(1)
                    print(f"Run_id trouvé : {run_id}")
                    return 0, run_id
        print(f"Erreur : run_id introuvable dans {self.log_file}")
        return 1, None


def create_tmp_yaml(step: Step, run_id):

    tmp_config = tmp_dir / "tmp.yml"
    shutil.copyfile(step.config, tmp_config)
    with open(tmp_config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})["run_id"] = run_id
    with open(tmp_config, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"tmp.yaml créé avec run_id {run_id}")
    return tmp_config


# ---------- Étape 1 ----------

def run_train(step: Step):
    try:
        print(f"Lancement de {step.name} ... (log -> {step.log_file})")
        with open(step.log_file, "w") as f:
            subprocess.run([PYTHON, step.script, "--yaml", step.config],
                           stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        if step.returns_run_id:
            return step._extract_run_id()

        return 0

    except subprocess.CalledProcessError:
        print(
            f"Erreur lors de l'exécution de mnist.py (voir {step.log_file})")
        return 1, None

# ---------- Étape 2 ----------


def run_train_again(step: Step, run_id):
    try:
        print(f"Lancement de {step.name} ... (log -> {step.log_file})")
        with open(step.log_file, "w") as f:
            subprocess.run([PYTHON, step.script, "--run-id",  run_id] + step.extra_args,
                           stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        return 0
    except subprocess.CalledProcessError:
        print(f"Erreur lors de l'exécution de mnist.py (voir {step.log_file})")
        return 1


# ---------- Étape 3 ----------


def run_mnist_infer(step: Step, run_id):
    try:
        print(f"Lancement de {step.name} ... (log -> {step.log_file})")

        tmp_config = create_tmp_yaml(step, run_id=run_id)
        with open(step.log_file, "w") as f:
            subprocess.run([PYTHON, step.script, "--yaml", tmp_config],
                           stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        return 0
    except subprocess.CalledProcessError:
        print(
            f"Erreur lors de l'exécution de mnist_infer.py (voir {step.log_file})")
        return 1


# ---------- Étape 4 ----------

def run_mlflow_infer(step: Step, run_id):
    try:
        print(f"Lancement de {step.name} ... (log -> {step.log_file})")
        with open(step.log_file, "w") as f:
            subprocess.run([PYTHON, step.script, "--run-id", run_id] + step.extra_args,
                           stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        return 0
    except subprocess.CalledProcessError:
        print(
            f"Erreur lors de l'exécution de mlflow_infer.py (voir {step.log_file})")
        return 1


# ---------- Main ----------


def main():

    os.makedirs(tmp_dir, exist_ok=True)

    steps = [
        Step(
            name="mnist_train",
            func=run_train,
            script=root_dir / "mnist_train.py",
            config=root_dir / "config_mnist.yml",
            log_file=tmp_dir / "mnist.log",
            returns_run_id=True
        ),
        Step(
            name="mnist_train_again",
            func=run_train_again,
            script=root_dir / "mnist_train.py",
            config=root_dir / "config_mnist.yml",
            log_file=tmp_dir / "mnist_again.log",
            needs_run_id=True,
            extra_args=["--epochs", "4"]
        ),
        Step(
            name="mnist_infer",
            func=run_mnist_infer,
            script=root_dir / "mnist_infer.py",
            config=root_dir / "config_infer.yml",
            log_file=tmp_dir / "mnist_infer.log",
            needs_run_id=True
        ),
        Step(
            name="mnist_mlflow_infer",
            func=run_mlflow_infer,
            script=root_dir / "mlflow_infer.py",
            log_file=tmp_dir / "mnist_mlflow_infer.log",
            needs_run_id=True
        ),
        Step(
            name="wgan_train",
            func=run_train,
            script=root_dir / "wgan_train.py",
            config=root_dir / "config_wgan.yml",
            log_file=tmp_dir / "wgan_train.log",
            returns_run_id=True
        ),
        Step(name="wgan_again",
             func=run_train_again,
             script=root_dir / "wgan_train.py",
             config=root_dir / "config_wgan.yml",
             log_file=tmp_dir / "wgan_again.log",
             needs_run_id=True,
             extra_args=["--epochs", "20"]
             ),
        Step(
            name="wgan_mlflow_infer",
            func=run_mlflow_infer,
            script=root_dir / "wgan_infer.py",
            log_file=tmp_dir / "wgan_infer.log",
            needs_run_id=True,
            extra_args=["--output", str(tmp_dir / "wgan.jpg")]
        ),
        Step(
            name="ddpm_train",
            func=run_train,
            script=root_dir / "ddpm_train.py",
            config=root_dir / "config_ddpm.yml",
            log_file=tmp_dir / "ddpm_train.log",
            returns_run_id=True
        ),
        Step(
            name="ddpm_mlflow_infer",
            func=run_mlflow_infer,
            script=root_dir / "ddpm_infer.py",
            log_file=tmp_dir / "ddpm_infer.log",
            needs_run_id=True,
            extra_args=["--output", str(tmp_dir / "ddpm.jpg")]
        ),
        Step(
            name="mnist_find_lr",
            func=run_train,
            script=root_dir / "mnist_find_lr.py",
            config=root_dir / "config_mnist.yml",
            log_file=tmp_dir / "mnist_find_lr.log",
            returns_run_id=False,
            needs_run_id=False
        )
    ]

    for step, action in zip(steps, actions):
        step.status = StepStatus.TEST if action else StepStatus.SKIP

    run_id = None

    for step in steps:
        if step.status != StepStatus.TEST:
            print(f"skip {step.name}")
            continue
        if step.needs_run_id:
            code = step.func(step, run_id)
        else:
            result = step.func(step)
            code, run_id = result if step.returns_run_id else (result, run_id)

        step.status = StepStatus.OK if code == 0 else StepStatus.FAIL

        if code != 0:
            break

    print("\n=== Résumé des étapes ===")
    for step in steps:
        print(f"{step.name}: {step.status.name}")


if __name__ == "__main__":
    import numpy as np
    from create_sindata import sine_data_generation
    from pathlib import Path

    file_path = Path(__file__).parent / "data" / "sin_wave.csv"

    # Crée le dossier si nécessaire
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Test d'existence
    if not os.path.exists(file_path):
        data = sine_data_generation(1000, 600, 60, 0.05)
        np.savetxt(file_path, data, delimiter=",")
        print(f"Fichier créé : {file_path}")
    else:
        print(f"Le fichier existe déjà : {file_path}")

    main()
