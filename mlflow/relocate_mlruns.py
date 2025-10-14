import yaml
import os
import re


def retrieve_yaml(dir):
    yaml_file = None
    runs = []
    for f in os.listdir(dir):
        fullname = os.path.join(dir, f)
        if ".yaml" in f:
            yaml_file = fullname
        elif os.path.isdir(fullname):
            runs.append(f)
    return yaml_file, runs


def replace_inpath(f, meta, out_path, key="artifact_location"):
    location = meta[key]
    pattern = re.compile(f"file://(/.*)/{f}")
    m = pattern.match(location)
    if m:
        in_path = m.group(1)
        meta[key] = location.replace(in_path, out_path)
    else:
        print(f"original path not found in {location}, no change")


def relocate_mlruns(out_path):
    for f in os.listdir(out_path):
        if f == ".trash":
            continue
        expe_dir = os.path.join(out_path, f)
        yaml_file, runs = retrieve_yaml(expe_dir)
        if yaml_file is not None:
            with open(yaml_file, "r") as file:
                meta = yaml.safe_load(file)
                replace_inpath(f, meta, out_path)
                print(f"{meta['name']} ({f})")

            with open(yaml_file, "w") as file:
                yaml.safe_dump(meta, file)

            print(f".. {len(runs)} runs à traiter")
            for r in runs:
                run_dir = os.path.join(expe_dir, r)
                yaml_file, _ = retrieve_yaml(run_dir)
                with open(yaml_file, "r") as file:
                    meta = yaml.safe_load(file)
                    deleted = meta["lifecycle_stage"] == "deleted"
                    if not deleted:
                        replace_inpath(f, meta, out_path, key="artifact_uri")
                        print(f"... {meta['run_name']} ({r})")

                        with open(yaml_file, "w") as file:
                            yaml.safe_dump(meta, file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Relocate artifact paths in mlruns directory.")
    parser.add_argument('mlruns', help="Path to mlruns root directory")

    args = parser.parse_args()

    relocate_mlruns(args.mlruns)
