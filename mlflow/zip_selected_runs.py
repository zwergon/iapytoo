import os
import yaml
import zipfile
from pathlib import Path


def retrieve_meta_yaml(path):
    """Charge un fichier meta.yaml si présent."""
    meta_file = os.path.join(path, "meta.yaml")
    if os.path.isfile(meta_file):
        with open(meta_file, "r") as f:
            return yaml.safe_load(f)
    return None


def should_include_run(meta, run_ids, tag_filters=None):
    """Détermine si un run est à inclure selon son ID et ses tags."""
    if not meta or meta.get("lifecycle_stage") == "deleted":
        return False

    if meta.get("run_id") not in run_ids:
        return False

    if tag_filters:
        tags = {tag["key"]: tag["value"] for tag in meta.get("tags", [])}
        for k, v in tag_filters.items():
            if tags.get(k) != v:
                return False

    return True


def add_to_zip(zipf, root_path, arc_base=""):
    """Ajoute tous les fichiers de root_path à zipf en préservant l’arborescence."""
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, start=arc_base)
            zipf.write(full_path, arcname=rel_path)


def zip_runs_tree(mlruns_root, run_ids, zip_output_path, tag_filters=None):
    mlruns_root = Path(mlruns_root)
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for experiment_dir in mlruns_root.iterdir():
            if not experiment_dir.is_dir() or experiment_dir.name == ".trash":
                continue

            # Inclure le meta.yaml de l'expérience s’il existe
            meta_path = experiment_dir / "meta.yaml"
            if meta_path.exists():
                zipf.write(meta_path, arcname=os.path.relpath(
                    meta_path, mlruns_root))

            # Vérifie tous les runs
            for run_dir in experiment_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                meta = retrieve_meta_yaml(run_dir)
                if should_include_run(meta, run_ids, tag_filters):
                    add_to_zip(zipf, run_dir, arc_base=mlruns_root)


# Exemple d'utilisation
if __name__ == "__main__":
    mlruns_root = "/work/lecomtje/Repositories/iapy/iapytoo/examples/mlruns"
    output_zip = "/work/lecomtje/Repositories/iapy/iapytoo/examples/zipped_runs/selected_runs.zip"

    run_ids = [
        "5d3c17b0043249d69dd44b4d79205127",
        "223e3cc70f1d49cb967024d05de3fa95"
    ]
    tag_filters = {
        # ex: "model": "v2"
        # laisser vide pour ne pas filtrer par tags
    }

    zip_runs_tree(mlruns_root, run_ids, output_zip, tag_filters)
    print(f"✅ Archive créée : {output_zip}")
