import subprocess
import re
import yaml
import os
import shutil
import sys
from pathlib import Path

PYTHON = sys.executable
root_dir = Path(__file__).parent

tmp_dir = root_dir / 'tmp'


mnist_script = root_dir / "mnist.py"
mnist_config = root_dir / "config_mnist.yml"

infer_config_template = root_dir / "config_infer.yml"
tmp_infer_config = tmp_dir / "tmp.yml"

mnist_infer_script = root_dir / "mnist_infer.py"
mlflow_infer_script = root_dir / "mlflow_infer.py"

# ---------- Étape 1 ----------


def run_mnist():
    log_file = tmp_dir / "mnist.log"
    try:
        print("Lancement de mnist.py ... (log -> mnist.log)")
        with open(log_file, "w") as f:
            result = subprocess.run([PYTHON, mnist_script, "--yaml", mnist_config],
                                    stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        # lire le log pour extraire run_id
        with open(log_file, "r") as f:
            for line in f:
                match = re.search(r"Run_id:\s*([a-f0-9]+)", line)
                if match:
                    run_id = match.group(1)
                    print(f"Run_id trouvé : {run_id}")
                    return 0, run_id
        print("Erreur : run_id introuvable dans mnist.log")
        return 1, None
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de mnist.py (voir {log_file})")
        return 1, None

# ---------- Étape 1 bis----------


def run_mnist_again(run_id):
    log_file = tmp_dir / "mnist_again.log"
    try:
        print("Lancement de mnist.py ... (log -> mnist_again.log)")
        with open(log_file, "w") as f:
            result = subprocess.run([PYTHON, mnist_script, "--run-id",  run_id, "--epochs", "4"],
                                    stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de mnist.py (voir {log_file})")
        return 1

# ---------- Étape 2 ----------


def create_tmp_yaml(run_id):
    try:
        shutil.copyfile(infer_config_template, tmp_infer_config)
        with open(tmp_infer_config, "r") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("model", {})["run_id"] = run_id
        with open(tmp_infer_config, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"tmp.yaml créé avec run_id {run_id}")
        return 0
    except Exception as e:
        print(f"Erreur lors de la création de tmp.yaml : {e}")
        return 1

# ---------- Étape 3 ----------


def run_mnist_infer():
    log_file = tmp_dir / "mnist_infer.log"
    try:
        print("Lancement de mnist_infer.py ... (log -> mnist_infer.log)")
        with open(log_file, "w") as f:
            subprocess.run([PYTHON, mnist_infer_script, "--yaml", tmp_infer_config],
                           stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(
            f"Erreur lors de l'exécution de mnist_infer.py (voir {log_file})")
        return 1

# ---------- Étape 4 ----------


def run_mlflow_infer(run_id):
    log_file = tmp_dir / "mlflow_infer.log"
    try:
        print("Lancement de mlflow_infer.py ... (log -> mlflow_infer.log)")
        with open(log_file, "w") as f:
            subprocess.run([PYTHON, mlflow_infer_script, run_id],
                           stdout=f, stderr=subprocess.STDOUT, text=True, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(
            f"Erreur lors de l'exécution de mlflow_infer.py (voir {log_file})")
        return 1

# ---------- Main ----------


def main():
    status = {}

    os.makedirs(tmp_dir, exist_ok=True)

    code, run_id = run_mnist()
    status['run_mnist'] = "OK" if code == 0 else "FAIL"

    if code == 0:
        code = run_mnist_again(run_id)
    status['run_mnist_again'] = "OK" if code == 0 else "FAIL"

    if code == 0:
        code = create_tmp_yaml(run_id)
    status['create_tmp_yaml'] = "OK" if code == 0 else "FAIL"

    if code == 0:
        code = run_mnist_infer()
    status['run_mnist_infer'] = "OK" if code == 0 else "FAIL"

    if code == 0:
        code = run_mlflow_infer(run_id)
    status['run_mlflow_infer'] = "OK" if code == 0 else "FAIL"

    print("\n=== Résumé des étapes ===")
    for k, v in status.items():
        print(f"{k}: {v}")
    print("\nLogs disponibles : mnist.log, mnist_infer.log, mlflow_infer.log")


if __name__ == "__main__":
    main()
