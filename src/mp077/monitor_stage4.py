from pathlib import Path
import argparse
import hashlib

import mlflow
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import mean_absolute_error

from mp077.run_stage4 import (
    ROOT,
    DB,
    REGISTERED_NAME,
    FEATURES,
    TARGET,
    MIN_IMPROVEMENT,
    validate_data,
    prepare_split,
)


NUMERIC_FEATURES = [
    "flu_cases_reported",
    "temp_c",
    "nearby_clinics",
    "discount_pct",
    "stock_lines",
    "distance_to_hospital_km",
]

LOCALITIES = [
    "Old Town",
    "New Town",
    "Cantonment",
]

PSI_BINS = 10
EPSILON = 0.0001

FINDINGS_PATH = ROOT / "reports/findings.yaml"


def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def calculate_psi(
    expected,
    actual,
    bins=PSI_BINS,
    epsilon=EPSILON
):

    expected = np.asarray(
        expected,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    assert len(expected) > 0
    assert len(actual) > 0

    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all()

    if expected.min() == expected.max():
        raise ValueError(
            "Constant reference feature requires "
            "special PSI handling."
        )

    # Create bins using the reference distribution.

    edges = np.linspace(
        expected.min(),
        expected.max(),
        bins + 1
    )

    # Include observations outside training range.

    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts = np.histogram(
        expected,
        bins=edges
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=edges
    )[0]

    assert expected_counts.sum() == len(expected)
    assert actual_counts.sum() == len(actual)

    expected_pct = (
        expected_counts / len(expected)
    )

    actual_pct = (
        actual_counts / len(actual)
    )

    expected_safe = np.maximum(
        expected_pct,
        epsilon
    )

    actual_safe = np.maximum(
        actual_pct,
        epsilon
    )

    psi = np.sum(
        (actual_safe - expected_safe)
        * np.log(actual_safe / expected_safe)
    )

    assert np.isfinite(psi)
    assert psi >= 0

    return float(psi)


def drift_status(psi):

    if psi < 0.10:
        return "Insignificant"

    if psi < 0.25:
        return "Moderate"

    return "Significant"


def monitor_drift(train, later):

    print("\n===== PSI DRIFT MONITORING =====")

    psi_results = {}

    for feature in NUMERIC_FEATURES:

        psi = calculate_psi(
            train[feature],
            later[feature]
        )

        psi_results[feature] = psi

        print(
            f"{feature:25s} "
            f"{psi:.6f} "
            f"{drift_status(psi)}"
        )

    most_drifted = max(
        psi_results,
        key=psi_results.get
    )

    print(
        "\nMost drifted feature:",
        most_drifted
    )

    print(
        "Maximum PSI:",
        psi_results[most_drifted]
    )

    return psi_results, most_drifted


def monitor_v1_by_locality(v1, later):

    print("\n===== V1 LOCALITY-WISE MONITORING =====")

    monitored = later.copy()

    monitored["prediction"] = v1.predict(
        monitored[FEATURES]
    )

    monitored["absolute_error"] = (
        monitored[TARGET]
        - monitored["prediction"]
    ).abs()

    assert np.isfinite(
        monitored["prediction"]
    ).all()

    assert set(
        monitored["locality"].unique()
    ) == set(LOCALITIES)

    mae_by_group = {}

    for locality in LOCALITIES:

        group = monitored[
            monitored["locality"] == locality
        ]

        mae = mean_absolute_error(
            group[TARGET],
            group["prediction"]
        )

        mae_by_group[locality] = float(mae)

        print(
            f"{locality:15s} "
            f"Records: {len(group):3d} "
            f"MAE: {mae:.6f}"
        )

    worst_group = max(
        mae_by_group,
        key=mae_by_group.get
    )

    overall_mae = mean_absolute_error(
        monitored[TARGET],
        monitored["prediction"]
    )

    print("\nOverall V1 later MAE:", overall_mae)
    print("Worst served group:", worst_group)

    return mae_by_group, worst_group


def compare_registered_models(
    v1,
    v2,
    train,
    later
):

    _, _, X_check, y_check = prepare_split(
        train,
        later
    )

    p1 = v1.predict(X_check)
    p2 = v2.predict(X_check)

    assert np.isfinite(p1).all()
    assert np.isfinite(p2).all()

    mae_v1 = float(
        mean_absolute_error(y_check, p1)
    )

    mae_v2 = float(
        mean_absolute_error(y_check, p2)
    )

    assert mae_v1 > 0

    improvement = (
        (mae_v1 - mae_v2) / mae_v1
    ) * 100

    eligible = (
        mae_v2 <= mae_v1 * (1 - MIN_IMPROVEMENT)
    )

    decision = "promote" if eligible else "keep"

    print("\n===== REGISTERED MODEL COMPARISON =====")

    print("V1 check MAE:", mae_v1)
    print("V2 check MAE:", mae_v2)
    print("MAE improvement percentage:", improvement)
    print("Decision:", decision)

    return mae_v1, mae_v2, decision


def update_findings(
    monitoring,
    champion_version
):

    print("\n===== UPDATE FINDINGS =====")

    with FINDINGS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:

        findings = yaml.safe_load(file)

    assert findings["group_id"] == "G067"
    assert findings["problem_id"] == "MP077"

    assert findings["tracking"][
        "registered_model_name"
    ] == REGISTERED_NAME

    original_data_issues = findings[
        "data_issues"
    ].copy()

    original_best_run = findings[
        "tracking"
    ]["best_run_id"]

    original_experiment = findings[
        "tracking"
    ]["experiment_name"]

    assert set(findings["monitoring"]) == set(
        monitoring
    )

    findings["monitoring"] = monitoring

    findings["tracking"][
        "champion_version"
    ] = int(champion_version)

    temporary_path = FINDINGS_PATH.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            findings,
            file,
            sort_keys=False,
            allow_unicode=True
        )

    with temporary_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        verified = yaml.safe_load(file)

    assert verified["data_issues"] == original_data_issues

    assert verified["tracking"][
        "best_run_id"
    ] == original_best_run

    assert verified["tracking"][
        "experiment_name"
    ] == original_experiment

    assert verified["monitoring"] == monitoring

    assert verified["tracking"][
        "champion_version"
    ] == int(champion_version)

    temporary_path.replace(FINDINGS_PATH)

    print("Findings file:", FINDINGS_PATH)
    print("Original data issues preserved: True")
    print("Stage 2 experiment preserved: True")
    print("Stage 2 best run preserved: True")
    print("Champion version updated:", champion_version)
    print("Monitoring findings saved: True")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--write",
        action="store_true",
        help="Update reports/findings.yaml"
    )

    args = parser.parse_args()

    assert Path.cwd().resolve() == ROOT.resolve(), (
        "Run this script from the repository root."
    )

    mlflow.set_tracking_uri(DB)
    mlflow.set_registry_uri(DB)

    client = mlflow.MlflowClient()

    protected_paths = [
        ROOT / "mlflow.db",
        ROOT / "models/model.joblib",
        ROOT / "data/raw/train.csv",
        ROOT / "data/raw/later.csv",
        ROOT / "data/raw/holdout_features.csv",
        ROOT / "predictions/holdout_predictions.csv",
    ]

    original_hashes = {
        path: sha256(path)
        for path in protected_paths
    }

    findings_hash_before = sha256(
        FINDINGS_PATH
    )

    print("===== LOAD REGISTERED MODELS =====")

    v1 = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_NAME}/1"
    )

    v2 = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_NAME}/2"
    )

    assert list(v1.feature_names_in_) == FEATURES
    assert list(v2.feature_names_in_) == FEATURES

    champion = client.get_model_version_by_alias(
        REGISTERED_NAME,
        "champion"
    )

    print("V1 loading: PASS")
    print("V2 loading: PASS")
    print("Current champion:", champion.version)

    train, later = validate_data()

    psi, most_drifted = monitor_drift(
        train,
        later
    )

    mae_by_group, worst_group = (
        monitor_v1_by_locality(v1, later)
    )

    mae_v1, mae_v2, decision = (
        compare_registered_models(
            v1,
            v2,
            train,
            later
        )
    )

    expected_champion = (
        "2" if decision == "promote" else "1"
    )

    assert str(
        champion.version
    ) == expected_champion

    monitoring = {
        "psi": psi,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "decision": decision,
    }

    print("\n===== MONITORING RESULTS =====")

    print(
        yaml.safe_dump(
            monitoring,
            sort_keys=False
        )
    )

    for path, old_hash in original_hashes.items():

        assert sha256(path) == old_hash, (
            f"Unexpected modification: {path}"
        )

    if args.write:

        update_findings(
            monitoring,
            champion.version
        )

    else:

        assert sha256(
            FINDINGS_PATH
        ) == findings_hash_before

        print("Dry run: findings.yaml unchanged")

    print("\n===== FINAL SAFETY VERIFICATION =====")

    print("MLflow database unchanged: True")
    print("Exported model unchanged: True")
    print("Raw datasets unchanged: True")
    print("Holdout predictions unchanged: True")
    print("Champion decision verified: True")
    print("No retraining performed")
    print("Monitoring completed successfully")


if __name__ == "__main__":
    main()