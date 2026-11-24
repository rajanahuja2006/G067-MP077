from mlflow.models import infer_signature
import hashlib
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from mp077.run_stage2 import clean


ROOT = Path(__file__).resolve().parents[2]

DB = "sqlite:///mlflow.db"
EXPERIMENT = "mp077-stage4"
REGISTERED_NAME = "mp077-model"

TARGET = "units_sold"

FEATURES = [
    "flu_cases_reported",
    "temp_c",
    "nearby_clinics",
    "discount_pct",
    "stock_lines",
    "distance_to_hospital_km",
    "shop_type",
    "locality",
]

TRAIN_PATH = ROOT / "data/raw/train.csv"
LATER_PATH = ROOT / "data/raw/later.csv"

EXPECTED_TRAIN_SHA = (
    "8d00fd3a8a17d19bbf8db246e5047df5d769eb6af22f54fee0012748ffc034a3"
)

EXPECTED_LATER_SHA = (
    "d5f724cc4e81389400287d6759330a760763d1bdcd0bb61db331044f34633430"
)

EXPECTED_V1_MAE = 153.7802848523837
EXPECTED_V2_MAE = 90.4721098804522

MIN_IMPROVEMENT = 0.10


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_data():

    assert sha256(TRAIN_PATH) == EXPECTED_TRAIN_SHA
    assert sha256(LATER_PATH) == EXPECTED_LATER_SHA

    train = clean(
        pd.read_csv(TRAIN_PATH),
        dedupe=True
    )

    later = pd.read_csv(LATER_PATH)

    required = set(
        FEATURES + [TARGET, "record_id", "date"]
    )

    assert set(train.columns) == required
    assert set(later.columns) == required

    assert len(train) == 1000
    assert len(later) == 300

    assert not train.isna().any().any()
    assert not later.isna().any().any()

    assert train["record_id"].is_unique
    assert later["record_id"].is_unique

    assert set(train["record_id"]).isdisjoint(
        later["record_id"]
    )

    print("Dataset fingerprints verified")
    print("Cleaned train rows:", len(train))
    print("Later rows:", len(later))

    return train, later


def prepare_split(train, later):

    train = train.copy()
    later = later.copy()

    train["date"] = pd.to_datetime(train["date"])
    later["date"] = pd.to_datetime(later["date"])

    later = later.sort_values(
        ["date", "record_id"],
        kind="stable"
    ).reset_index(drop=True)

    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    check = later.iloc[split_index:].copy()

    combined = pd.concat(
        [train, later_train],
        ignore_index=True
    )

    assert len(later_train) == 210
    assert len(check) == 90
    assert len(combined) == 1210

    assert combined["record_id"].is_unique

    assert set(combined["record_id"]).isdisjoint(
        check["record_id"]
    )

    assert train["date"].max() < later["date"].min()

    assert later_train["date"].max() <= check["date"].min()

    X_train = combined[FEATURES]
    y_train = combined[TARGET]

    X_check = check[FEATURES]
    y_check = check[TARGET]

    print("\n===== CHRONOLOGICAL SPLIT =====")

    print("Later training rows:", len(later_train))
    print("Combined training rows:", len(combined))
    print("Check rows:", len(check))

    print(
        "Training date range:",
        combined["date"].min(),
        "to",
        combined["date"].max()
    )

    print(
        "Check date range:",
        check["date"].min(),
        "to",
        check["date"].max()
    )

    return X_train, y_train, X_check, y_check


def train_v2(v1, X_train, y_train):

    # Clone creates an unfitted pipeline.
    # Original V1 remains unchanged.

    v2 = clone(v1)

    v2.fit(X_train, y_train)

    print("\n===== V2 TRAINING =====")

    print(
        "Algorithm:",
        type(v2.named_steps["model"]).__name__
    )

    print("Training completed")
    print("Training records:", len(X_train))

    return v2


def evaluate_models(
    v1,
    v2,
    X_train,
    y_train,
    X_check,
    y_check
):

    pred_v1 = v1.predict(X_check)
    pred_v2 = v2.predict(X_check)

    assert np.isfinite(pred_v1).all()
    assert np.isfinite(pred_v2).all()

    mae_v1 = mean_absolute_error(
        y_check,
        pred_v1
    )

    mae_v2 = mean_absolute_error(
        y_check,
        pred_v2
    )

    rmse_v1 = np.sqrt(
        mean_squared_error(y_check, pred_v1)
    )

    rmse_v2 = np.sqrt(
        mean_squared_error(y_check, pred_v2)
    )

    train_mae_v2 = mean_absolute_error(
        y_train,
        v2.predict(X_train)
    )

    assert mae_v1 > 0

    improvement = (
        (mae_v1 - mae_v2) / mae_v1
    )

    threshold = mae_v1 * (
        1 - MIN_IMPROVEMENT
    )

    promote = mae_v2 <= threshold

    assert np.isclose(
        mae_v1,
        EXPECTED_V1_MAE,
        rtol=1e-6
    )

    assert np.isclose(
        mae_v2,
        EXPECTED_V2_MAE,
        rtol=1e-6
    )

    print("\n===== MODEL COMPARISON =====")

    print("V1 check MAE:", mae_v1)
    print("V2 check MAE:", mae_v2)

    print("V1 check RMSE:", rmse_v1)
    print("V2 check RMSE:", rmse_v2)

    print("V2 training MAE:", train_mae_v2)

    print("Maximum allowed V2 MAE:", threshold)

    print(
        "Improvement percentage:",
        improvement * 100
    )

    print("Required improvement: 10%")
    print("Promotion criterion satisfied:", promote)

    print(
        "Proposed decision:",
        "promote" if promote else "keep"
    )

    return {
        "mae_v1_on_check": float(mae_v1),
        "mae_v2_on_check": float(mae_v2),
        "rmse_v1_on_check": float(rmse_v1),
        "rmse_v2_on_check": float(rmse_v2),
        "train_mae_v2": float(train_mae_v2),
        "improvement_pct": float(improvement * 100),
        "promote": bool(promote),
    }


def register_v2(
    client,
    v2,
    X_train,
    X_check,
    results
):

    print("\n===== PRE-REGISTRATION CHECK =====")

    versions = client.search_model_versions(
        "name = 'mp077-model'"
    )

    assert len(versions) == 1
    assert str(versions[0].version) == "1"

    champion = client.get_model_version_by_alias(
        REGISTERED_NAME,
        "champion"
    )

    assert str(champion.version) == "1"

    experiment = client.get_experiment_by_name(
        EXPERIMENT
    )

    if experiment is None:

        experiment_id = client.create_experiment(
            name=EXPERIMENT,
            artifact_location="mlruns/stage4"
        )

    else:

        experiment_id = experiment.experiment_id

        existing_runs = client.search_runs(
            experiment_ids=[experiment_id]
        )

        if existing_runs:
            raise RuntimeError(
                "Stage 4 runs already exist. "
                "Inspect them before registering again."
            )

    print("Experiment:", EXPERIMENT)
    print("Experiment ID:", experiment_id)

    print("\n===== MLFLOW V2 LOGGING =====")

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="linear-regression-v2"
    ) as run:

        run_id = run.info.run_id

        mlflow.set_tags({
            "stage": "4",
            "data_sha256": sha256(TRAIN_PATH),
            "data_sha": sha256(TRAIN_PATH),
            "later_data_sha256": sha256(LATER_PATH),
            "parent_model_version": "1",
            "parent_run_id":
                "22ac88d2bb0b43e18302e81f02348a03",
            "purpose": "drift_retraining",
            "promotion_eligible":
                str(results["promote"]).lower(),
        })

        mlflow.log_params({
            "algorithm": "LinearRegression",
            "original_train_rows": 1000,
            "later_train_rows": 210,
            "combined_train_rows": len(X_train),
            "check_rows": len(X_check),
            "later_train_fraction": 0.70,
            "later_check_fraction": 0.30,
            "minimum_improvement_pct": 10,
            "split_method": "chronological",
            "split_sort_columns": "date,record_id",
            "feature_count": len(FEATURES),
        })

        mlflow.log_metrics({
            "mae": results["mae_v2_on_check"],
            "mae_v1_on_check":
                results["mae_v1_on_check"],
            "mae_v2_on_check":
                results["mae_v2_on_check"],
            "rmse_v1_on_check":
                results["rmse_v1_on_check"],
            "rmse_v2_on_check":
                results["rmse_v2_on_check"],
            "train_mae":
                results["train_mae_v2"],
            "mae_improvement_pct":
                results["improvement_pct"],
            "generalization_gap":
                results["mae_v2_on_check"]
                - results["train_mae_v2"],
        })

        signature = infer_signature(
            X_train,
            v2.predict(X_train)
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=v2,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            serialization_format="cloudpickle"
        )

        print("Run ID:", run_id)
        print("Model URI:", model_info.model_uri)

        # Verify logged artifact before registration.

        logged_model = mlflow.sklearn.load_model(
            model_info.model_uri
        )

        assert np.allclose(
            logged_model.predict(X_check),
            v2.predict(X_check),
            rtol=1e-9,
            atol=1e-8
        )

        print("Logged model prediction verification: PASS")

    print("\n===== REGISTER MODEL V2 =====")

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_NAME
    )

    version = str(registered.version)

    assert version == "2"

    client.set_model_version_tag(
        REGISTERED_NAME,
        version,
        "stage",
        "4"
    )

    client.set_model_version_tag(
        REGISTERED_NAME,
        version,
        "data_sha256",
        sha256(TRAIN_PATH)
    )

    client.set_model_version_tag(
        REGISTERED_NAME,
        version,
        "promotion_eligible",
        str(results["promote"]).lower()
    )

    print("Registered model:", REGISTERED_NAME)
    print("Registered version:", version)

    print("\n===== REGISTERED V2 VERIFICATION =====")

    registered_v2 = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_NAME}/{version}"
    )

    expected = v2.predict(X_check)
    actual = registered_v2.predict(X_check)

    assert np.allclose(
        actual,
        expected,
        rtol=1e-9,
        atol=1e-8
    )

    print(
        "Maximum prediction difference:",
        np.max(np.abs(actual - expected))
    )

    print("Registered V2 predictions match: True")

    champion = client.get_model_version_by_alias(
        REGISTERED_NAME,
        "champion"
    )

    assert str(champion.version) == "1"

    print("\n===== REGISTRATION SUMMARY =====")

    print("V2 run ID:", run_id)
    print("V2 version:", version)
    print("V2 registration: SUCCESS")
    print("Champion version:", champion.version)
    print("Champion alias unchanged: True")

    return run_id, version


def main():

    assert Path.cwd().resolve() == ROOT, (
        "Execute from project root"
    )

    mlflow.set_tracking_uri(DB)
    mlflow.set_registry_uri(DB)

    client = mlflow.MlflowClient()

    model_path = ROOT / "models/model.joblib"
    db_path = ROOT / "mlflow.db"

    model_hash_before = sha256(model_path)
    db_hash_before = sha256(db_path)

    champion = client.get_model_version_by_alias(
        REGISTERED_NAME,
        "champion"
    )

    assert str(champion.version) == "1"

    versions = client.search_model_versions(
        "name = 'mp077-model'"
    )

    assert len(versions) == 1

    v1 = mlflow.sklearn.load_model(
        "models:/mp077-model/1"
    )

    assert list(v1.feature_names_in_) == FEATURES

    print("Original V1 loaded successfully")

    train, later = validate_data()

    X_train, y_train, X_check, y_check = (
        prepare_split(train, later)
    )

    original_predictions = v1.predict(X_check)

    v2 = train_v2(
        v1,
        X_train,
        y_train
    )

    results = evaluate_models(
        v1,
        v2,
        X_train,
        y_train,
        X_check,
        y_check
    )

    assert np.array_equal(
        original_predictions,
        v1.predict(X_check)
    )

    assert sha256(model_path) == model_hash_before
    assert sha256(db_path) == db_hash_before

    print("\n===== SAFETY VERIFICATION =====")

    print("Original V1 predictions unchanged: True")
    print("Original V1 model unchanged: True")
    print("MLflow database unchanged: True")
    print("Champion remains V1: True")
    print("V2 evaluation completed successfully")
    print("Ready for MLflow registration")

    print("\nResults:", results)

    run_id, version = register_v2(
        client,
        v2,
        X_train,
        X_check,
        results
    )

    assert sha256(model_path) == model_hash_before

    print("\n===== FINAL STAGE 4 SAFETY =====")

    print("V2 run ID:", run_id)
    print("Registered version:", version)
    print("Original V1 model unchanged: True")
    print("Champion remains V1: True")
    print("No Stage 2 retraining performed")
    print("Ready for conditional champion promotion")


if __name__ == "__main__":
    main()