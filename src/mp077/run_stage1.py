"""Run all seven checks and print the result as findings.yaml data_issues."""
import json

import pandas as pd
import yaml

from mp077.data_checks import run_all, sha256_of


def main(params_path="params.yaml"):
    with open(params_path) as f:
        p = yaml.safe_load(f)
    train = pd.read_csv(p["data"]["train"])
    later = pd.read_csv(p["data"]["later"])
    holdout = pd.read_csv(p["data"]["holdout"])
    print("train sha256:", sha256_of(p["data"]["train"]))
    print(json.dumps(run_all(train, later, holdout), indent=2, default=str))


if __name__ == "__main__":
    main()
