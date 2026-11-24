import hashlib
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from mp077.run_stage2 import clean

from mp077.run_stage4 import (
    ROOT,
    DB,
    REGISTERED_NAME,
    FEATURES,
    TARGET,
)


HOLDOUT_PATH = ROOT / "data/raw/holdout_features.csv"

OUTPUT_PATH = (
    ROOT / "predictions/holdout_predictions.csv"
)

EXPECTED_CHAMPION = "2"
EXPECTED_ROWS = 250

EXPECTED_RUN_ID = (
    "df8bc1f1d3b3421b9dbbbc7311460f45"
)


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def validate_holdout():

    raw = pd.read_csv(
        HOLDOUT_PATH,
        dtype={"record_id": "string"}
    )

    required_columns = set(
        FEATURES + ["record_id", "date"]
    )

    assert set(raw.columns) == required_columns

    assert len(raw) == EXPECTED_ROWS

    assert TARGET not in raw.columns

    assert raw["record_id"].is_unique

    assert not raw.isna().any().any()

    assert raw.duplicated().sum() == 0

    dates = pd.to_datetime(
        raw["date"],
        errors="raise"
    )

    assert dates.min() >= pd.Timestamp("2026-09-01")
    assert dates.max() <= pd.Timestamp("2026-09-30")

    print("===== HOLDOUT DATA VALIDATION =====")

    print("Holdout rows:", len(raw))
    print("Holdout columns:", len(raw.columns))
    print("Duplicate records:", raw.duplicated().sum())
    print("Missing values:", raw.isna().sum().sum())
    print("Target column present:", TARGET in raw.columns)

    print(
        "Date range:",
        dates.min(),
        "to",
        dates.max()
    )

    processed = clean(raw, dedupe=False)

    assert len(processed) == EXPECTED_ROWS

    assert processed["record_id"].equals(
        raw["record_id"]
    )

    assert not processed.isna().any().any()

    # Holdout is already clean.
    # Verify cleaning has not changed feature values.

    pd.testing.assert_frame_equal(
        raw[FEATURES],
        processed[FEATURES],
        check_dtype=False
    )

    X = processed[FEATURES].copy()

    assert X.shape == (250, 8)

    print("Input feature shape:", X.shape)
    print("Holdout validation: PASS")

    return raw, X


def main():

    assert Path.cwd().resolve() == ROOT

    mlflow.set_tracking_uri(DB)
    mlflow.set_registry_uri(DB)

    client = mlflow.MlflowClient()

    db_path = ROOT / "mlflow.db"
    exported_model = ROOT / "models/model.joblib"

    db_hash_before = sha256(db_path)

    exported_hash_before = sha256(
        exported_model
    )

    holdout_hash_before = sha256(
        HOLDOUT_PATH
    )

    print("===== CHAMPION VERIFICATION =====")

    champion = client.get_model_version_by_alias(
        REGISTERED_NAME,
        "champion"
    )

    assert str(champion.version) == EXPECTED_CHAMPION

    assert champion.run_id == EXPECTED_RUN_ID

    print("Registered model:", REGISTERED_NAME)
    print("Champion version:", champion.version)
    print("Champion run ID:", champion.run_id)

    model = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_NAME}@champion"
    )

    assert list(model.feature_names_in_) == FEATURES

    print("Champion model loading: PASS")

    raw, X = validate_holdout()

    print("\n===== GENERATE HOLDOUT PREDICTIONS =====")

    predictions = np.asarray(
        model.predict(X),
        dtype=float
    )

    assert predictions.shape == (EXPECTED_ROWS,)

    assert np.isfinite(predictions).all()

    output = pd.DataFrame({
        "record_id": raw["record_id"],
        "prediction": predictions
    })

    assert output.shape == (250, 2)

    assert output["record_id"].is_unique

    assert not output.isna().any().any()

    assert output["record_id"].equals(
        raw["record_id"]
    )

    print("Predictions generated:", len(output))

    print("\n===== PREDICTION STATISTICS =====")

    print(
        output["prediction"]
        .describe()
        .to_string()
    )

    print(
        "Negative predictions:",
        int((predictions < 0).sum())
    )

    print("\n===== FIRST 10 PREDICTIONS =====")

    print(output.head(10).to_string(index=False))

    print("\n===== SAVE PREDICTIONS =====")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".tmp"
    )

    output.to_csv(
        temporary_path,
        index=False
    )

    saved = pd.read_csv(
        temporary_path,
        dtype={"record_id": "string"}
    )

    assert list(saved.columns) == [
        "record_id",
        "prediction"
    ]

    assert saved.shape == (250, 2)

    assert saved["record_id"].equals(
        raw["record_id"]
    )

    assert np.allclose(
        saved["prediction"],
        predictions,
        rtol=1e-12,
        atol=1e-10
    )

    assert sha256(db_path) == db_hash_before

    assert sha256(exported_model) == exported_hash_before

    assert sha256(HOLDOUT_PATH) == holdout_hash_before

    temporary_path.replace(OUTPUT_PATH)

    print("Output file:", OUTPUT_PATH)

    print("\n===== FINAL VALIDATION =====")

    print("Output rows:", len(saved))
    print("Output columns:", saved.columns.tolist())
    print("Unique record IDs:", saved["record_id"].nunique())
    print("Missing values:", saved.isna().sum().sum())

    print("Record ID order preserved: True")
    print("All predictions finite: True")
    print("Saved predictions verified: True")
    print("MLflow database unchanged: True")
    print("Original exported V1 unchanged: True")
    print("Raw holdout unchanged: True")
    print("Champion version:", champion.version)
    print("No retraining performed")
    print("Holdout prediction generation: SUCCESS")


if __name__ == "__main__":
    main()