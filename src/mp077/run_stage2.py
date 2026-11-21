"""Stage 2 (Model lead): train, track in MLflow, register, predict on holdout.

Usage (from repo root):
  PYTHONPATH=src python3 -m mp077.run_stage2 train   --train data/train_clean.csv --later data/raw/later.csv
  PYTHONPATH=src python3 -m mp077.run_stage2 predict --holdout data/holdout.csv --out predictions.csv
Then:  mlflow ui --backend-store-uri sqlite:///mlflow.db   (screenshot the run table)
"""
import argparse
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

DB = "sqlite:///mlflow.db"          # P06: use a DB file, not ./mlruns
EXPERIMENT = "mp077-stage2"
REGISTERED_NAME = "mp077-model"
SEED = 42
LATER = None   # (X, y) from later.csv, set in train()


TARGET = "units_sold"
DROP = ["record_id", "date"]


def clean(df, dedupe=False):
    """Fix the issues found by the Stage-1 checks. Applied identically to every file."""
    df = df.copy()
    if dedupe:
        df = df.drop_duplicates()
    # spelling variants: lowercase, trim, unify separators (19 raw -> 3 true levels)
    st = (df["shop_type"].astype(str).str.strip().str.lower()
          .str.replace(r"[\s\-]+", "_", regex=True).str.rstrip("."))
    levels = ("chain", "hospital_store", "independent")
    st = st.map(lambda v: next((l for l in levels if l.startswith(v[:3])), v))  # 'cha.'->chain
    df = df.assign(shop_type=st)
    # unit mix-up: Old Town discount_pct is on a 0-1 scale instead of percent
    old = df["locality"].astype(str).str.strip().str.lower() == "old town"
    if old.any() and (~old).any():
        ratio = df.loc[old, "discount_pct"].mean() / df.loc[~old, "discount_pct"].mean()
        if ratio < 0.05:
            df.loc[old, "discount_pct"] = df.loc[old, "discount_pct"] * 100
    return df


def features(df):
    return df.drop(columns=[c for c in DROP + [TARGET] if c in df.columns])


def make_pipe(model, X):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                            remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])


def scores(pipe, X, y):
    p = pipe.predict(X)
    return mean_absolute_error(y, p), float(np.sqrt(mean_squared_error(y, p)))


def log_run(name, pipe, params, Xtr, ytr, Xte, yte, tags, cv_mae=None):
    with mlflow.start_run(run_name=name, tags=tags) as run:
        pipe.fit(Xtr, ytr)
        tr_mae, _ = scores(pipe, Xtr, ytr)
        te_mae, te_rmse = scores(pipe, Xte, yte)
        mlflow.log_params(params)
        mlflow.log_metrics({"train_mae": tr_mae, "mae": te_mae, "rmse": te_rmse,
                            "gap": te_mae - tr_mae})
        if LATER is not None:
            mlflow.log_metric("later_mae", mean_absolute_error(LATER[1], pipe.predict(LATER[0])))
        if cv_mae is not None:
            mlflow.log_metric("cv_mae", cv_mae)
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle",
                                 signature=infer_signature(Xte, pipe.predict(Xte)))
        print(f"{name:22s} train_mae={tr_mae:.3f} test_mae={te_mae:.3f} gap={te_mae-tr_mae:.3f}")
        return run.info.run_id, te_mae


def train(a):
    mlflow.set_tracking_uri(DB)
    mlflow.set_experiment(EXPERIMENT)
    df = clean(pd.read_csv(a.train), dedupe=True)
    print("shop_type levels:", sorted(df["shop_type"].unique()))
    X, y = features(df), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    global LATER
    if a.later:
        lt = clean(pd.read_csv(a.later))
        LATER = (features(lt), lt[TARGET])
    tags = {"stage": "2", "owner": "model-lead", "data_sha": a.data_sha}
    results = {}

    # P02: honest baseline
    results["baseline-mean"] = log_run("baseline-mean", make_pipe(DummyRegressor(), X),
        {"model": "mean"}, Xtr, ytr, Xte, yte, tags)
    # P03/P06: candidate models
    results["linear-baseline"] = log_run("linear-baseline", make_pipe(LinearRegression(), X),
        {"model": "linear"}, Xtr, ytr, Xte, yte, tags)
    for d in (2, 4, 6, 12):
        n = f"tree-depth-{d}"
        results[n] = log_run(n, make_pipe(DecisionTreeRegressor(max_depth=d, random_state=SEED), X),
            {"model": "tree", "max_depth": d}, Xtr, ytr, Xte, yte, {**tags, "sweep": "tree-depth"})
    # P03 T2: tune RF with 5-fold CV on TRAIN only (test set untouched)
    gs = GridSearchCV(make_pipe(RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1), X),
                      {"model__max_depth": [4, 8, 12, None]}, cv=5,
                      scoring="neg_mean_absolute_error").fit(Xtr, ytr)
    best_d = gs.best_params_["model__max_depth"]
    results["rf-tuned"] = log_run("rf-tuned", gs.best_estimator_,
        {"model": "random_forest", "max_depth": best_d, "n_estimators": 200},
        Xtr, ytr, Xte, yte, {**tags, "sweep": "rf-cv"}, cv_mae=-gs.best_score_)

    # Choose: lowest test MAE among non-baseline runs (swap for your own justified rule)
    cand = {k: v for k, v in results.items() if k != "baseline-mean"}
    best = min(cand, key=lambda k: cand[k][1])
    run_id, best_mae = cand[best]
    base_mae = results["baseline-mean"][1]
    print(f"\nWinner: {best} (MAE {best_mae:.3f}) beats mean baseline {base_mae:.3f} "
          f"by {base_mae-best_mae:.3f}")
    mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_NAME)
    mlflow.MlflowClient().set_registered_model_alias(REGISTERED_NAME, "champion", mv.version)
    print(f"Registered {REGISTERED_NAME} v{mv.version} @champion")


def predict(a):
    mlflow.set_tracking_uri(DB)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
    h = clean(pd.read_csv(a.holdout))
    ids = h["record_id"] if "record_id" in h.columns else h.index
    out = pd.DataFrame({"record_id": ids, "prediction": model.predict(features(h))})
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--train", required=True)
    t.add_argument("--later", default=None); t.add_argument("--data-sha", default="unknown")
    t.set_defaults(fn=train)
    p = sp.add_parser("predict"); p.add_argument("--holdout", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.set_defaults(fn=predict)
    a = ap.parse_args(); a.fn(a)