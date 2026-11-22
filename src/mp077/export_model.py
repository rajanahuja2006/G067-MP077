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
    import mlflow
    import sklearn
    from mp077.run_stage2 import clean, features

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    pipe = mlflow.sklearn.load_model("models:/mp077-model@champion")
    got = list(pipe.feature_names_in_)
    assert sorted(got) == sorted(FEATURES), f"unexpected features: {got}"
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL)

    h = features(clean(pd.read_csv("data/raw/holdout_features.csv"))).head(5)[FEATURES]
    h.to_csv(SAMPLE_IN, index=False)
    pd.DataFrame({"prediction": pipe.predict(h)}).to_csv(SAMPLE_OUT, index=False)
    INFO.write_text(json.dumps({"registered_model": "mp077-model", "alias": "champion",
                                "features": FEATURES, "sklearn": sklearn.__version__,
                                "pandas": pd.__version__, "numpy": np.__version__}, indent=2))
    print(f"Saved {MODEL} with {len(got)} input features. Now commit models/ and push.")


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
