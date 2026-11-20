"""Run all seven checks and print the result as findings.yaml data_issues."""
import json
from pathlib import Path

import pandas as pd
import yaml

from mp077.data_checks import run_all, sha256_of


def main(params_path="params.yaml"):
    with open(params_path) as f:
        p = yaml.safe_load(f)

    missing = [path for path in p["data"].values() if not Path(path).is_file()]
    if missing:
        paths = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing data file(s): {paths}. Restore DVC-tracked files with "
            "`dvc pull` (a DVC remote must first be configured)."
        )

    train = pd.read_csv(p["data"]["train"])
    later = pd.read_csv(p["data"]["later"])
    holdout = pd.read_csv(p["data"]["holdout"])
    print("train sha256:", sha256_of(p["data"]["train"]))
    print(json.dumps(run_all(train, later, holdout), indent=2, default=str))


if __name__ == "__main__":
    main()
