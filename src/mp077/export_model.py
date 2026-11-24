"""Export the registered champion (preprocessing + model) to models/model.joblib.

  PYTHONPATH=src python3 -m mp077.export_model export   # Stage 2 owner, once
  PYTHONPATH=src python3 -m mp077.export_model check    # run on ANY computer after git pull

`export` only READS the registry (no new version, `champion` alias is not moved).
`check` needs only: joblib, pandas, numpy, scikit-learn (same version as requirements.txt).

INPUT CONTRACT: 8 features, already clean --
  flu_cases_reported, temp_c, nearby_clinics, discount_pct (percent, not 0-1),
  stock_lines, distance_to_hospital_km, shop_type (chain|hospital_store|independent),
  locality (Cantonment|New Town|Old Town)
Do NOT pass record_id, date or units_sold.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL = Path("models/model.joblib")
SAMPLE_IN = Path("models/sample_input.csv")
SAMPLE_OUT = Path("models/sample_expected.csv")
INFO = Path("models/model_info.json")
FEATURES = ["flu_cases_reported", "temp_c", "nearby_clinics", "discount_pct", "stock_lines",
            "distance_to_hospital_km", "shop_type", "locality"]


def export():
    import hashlib
    import mlflow
    import sklearn

    from mp077.run_stage2 import clean, features

    db_path = Path("mlflow.db")

    db_hash_before = hashlib.sha256(
        db_path.read_bytes()
    ).hexdigest()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_registry_uri("sqlite:///mlflow.db")

    client = mlflow.MlflowClient()

    champion = client.get_model_version_by_alias(
        "mp077-model",
        "champion"
    )

    print("===== CHAMPION EXPORT =====")

    print("Registered model:", champion.name)
    print("Champion version:", champion.version)
    print("Champion run ID:", champion.run_id)

    pipe = mlflow.sklearn.load_model(
        "models:/mp077-model@champion"
    )

    got = list(pipe.feature_names_in_)

    assert got == FEATURES, (
        f"Unexpected feature order: {got}"
    )

    h = features(
        clean(
            pd.read_csv(
                "data/raw/holdout_features.csv"
            )
        )
    ).head(5)[FEATURES]

    assert h.shape == (5, 8)

    expected = np.asarray(
        pipe.predict(h),
        dtype=float
    )

    assert expected.shape == (5,)
    assert np.isfinite(expected).all()

    Path("models").mkdir(exist_ok=True)

    temporary_model = MODEL.with_suffix(".tmp")

    joblib.dump(pipe, temporary_model)

    restored = joblib.load(temporary_model)

    assert list(
        restored.feature_names_in_
    ) == FEATURES

    actual = restored.predict(h)

    assert np.allclose(
        expected,
        actual,
        rtol=1e-12,
        atol=1e-10
    )

    current_champion = (
        client.get_model_version_by_alias(
            "mp077-model",
            "champion"
        )
    )

    assert current_champion.version == champion.version
    assert current_champion.run_id == champion.run_id

    assert hashlib.sha256(
        db_path.read_bytes()
    ).hexdigest() == db_hash_before

    temporary_model.replace(MODEL)

    h.to_csv(
        SAMPLE_IN,
        index=False
    )

    pd.DataFrame({
        "prediction": expected
    }).to_csv(
        SAMPLE_OUT,
        index=False
    )

    model_hash = hashlib.sha256(
        MODEL.read_bytes()
    ).hexdigest()

    metadata = {
        "registered_model": "mp077-model",
        "alias": "champion",
        "champion_version": int(champion.version),
        "run_id": champion.run_id,
        "features": FEATURES,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "model_sha256": model_hash,
    }

    INFO.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\n===== EXPORT VERIFICATION =====")

    print("Exported model:", MODEL)
    print("Champion version:", champion.version)
    print("Input features:", len(got))
    print("Model SHA256:", model_hash)

    print(
        "Maximum prediction difference:",
        np.max(np.abs(expected - actual))
    )

    print("Sample predictions:", expected.tolist())

    print("MLflow database unchanged: True")
    print("Champion alias unchanged: True")
    print("No retraining performed")
    print("No new model registration performed")
    print("Champion export: SUCCESS")

def check():
    import sklearn
    info = json.loads(INFO.read_text())
    print(f"sklearn here={sklearn.__version__}  exported with={info['sklearn']}")
    pipe = joblib.load(MODEL)
    got = list(pipe.feature_names_in_)
    ok_feat = sorted(got) == sorted(FEATURES) and len(got) == 8
    X = pd.read_csv(SAMPLE_IN)
    pred = pipe.predict(X)
    ok_pred = np.allclose(pred, pd.read_csv(SAMPLE_OUT)["prediction"], rtol=1e-6)
    print(f"loads: OK | 8 features accepted: {ok_feat} | predictions match reference: {ok_pred}")
    print("sample predictions:", np.round(pred, 1).tolist())
    sys.exit(0 if ok_feat and ok_pred else 1)


if __name__ == "__main__":
    {"export": export, "check": check}[sys.argv[1]]()
