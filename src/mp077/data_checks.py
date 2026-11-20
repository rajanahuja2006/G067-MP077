"""Stage 1: detect the seven kinds of data problem in train.csv.

Every function takes the raw DataFrame and returns a plain dict that maps
straight into the matching block of findings.yaml.
"""
import hashlib

import pandas as pd

NUMERIC = ["flu_cases_reported", "temp_c", "nearby_clinics", "discount_pct",
           "stock_lines", "distance_to_hospital_km"]
TARGET = "units_sold"
SENTINELS = [-1, -9, -99, -999, -9999, 9, 99, 999, 9999]


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_leaking_column(train, holdout):
    """A column in train that is missing from holdout (other than the target)
    or that agrees almost exactly with the target."""
    extra = [c for c in train.columns if c not in holdout.columns and c != TARGET]
    corr = {c: abs(train[c].corr(train[TARGET])) for c in NUMERIC}
    suspicious = [c for c, r in corr.items() if r > 0.98]
    cols = extra + suspicious
    return {"present": bool(cols), "column": cols[0] if cols else None}


def check_unit_mixup(train, group_col="locality", ratio_min=5.0):
    """One group whose mean for a column is far from the others'.
    Returns the exact power-of-ten style factor (wrong = right x factor)."""
    known = [1000, 100, 60, 12, 10]
    for col in NUMERIC:
        means = train.groupby(group_col)[col].mean()
        for g in means.index:
            rest = train.loc[train[group_col] != g, col].mean()
            ratio = means[g] / rest if rest else float("inf")
            big = max(ratio, 1 / ratio) if ratio else float("inf")
            if big >= ratio_min:
                snapped = min(known, key=lambda k: abs(k - big) / k)
                factor = snapped if ratio > 1 else 1 / snapped
                return {"present": True, "column": col, "group_value": g,
                        "factor": factor, "observed_ratio": round(float(ratio), 4)}
    return {"present": False}


def check_sentinel(train):
    for col in NUMERIC + [TARGET]:
        vc = train[col].value_counts()
        for v in SENTINELS:
            n = int(vc.get(v, 0))
            # a sentinel is impossible AND repeated many times
            if n >= 10 and (v < 0 or v > train[col].quantile(0.99) * 3):
                return {"present": True, "column": col, "value": v, "count": n}
    return {"present": False}


def check_duplicates(train):
    return {"present": int(train.duplicated().sum()) > 0,
            "count": int(train.duplicated().sum())}


def check_stuck_sensor(train, min_days=7):
    """A column with one single value on every row for a run of >= min_days days."""
    d = train.drop_duplicates().sort_values("date")
    for col in NUMERIC:
        daily = d.groupby("date")[col].agg(["nunique", "first"]).reset_index()
        daily["const"] = daily["nunique"] == 1
        run = (daily["first"] != daily["first"].shift()) | (~daily["const"])
        daily["run"] = run.cumsum()
        g = daily[daily["const"]].groupby("run").agg(
            n=("date", "size"), start=("date", "min"), end=("date", "max"))
        g = g[g["n"] >= min_days]
        if len(g):
            r = g.sort_values("n").iloc[-1]
            return {"present": True, "column": col,
                    "start_date": r["start"], "end_date": r["end"]}
    return {"present": False}


def check_target_typos(train):
    """Values ~10x too big: a cluster above a wide empty gap in the sorted target."""
    s = train[TARGET].sort_values().reset_index(drop=True)
    gaps = s.diff()
    i = gaps.idxmax()
    big_gap = gaps[i] > 0.5 * s.iloc[i - 1] and s.iloc[i] > 3 * s.median()
    if big_gap:
        return {"present": True, "count": int(len(s) - i)}
    return {"present": False}


def check_spelling_variants(train, clean_reference):
    """Text columns whose raw spellings are more numerous than the clean levels
    in later.csv."""
    for col in train.select_dtypes("object").columns:
        if col in ("record_id", "date") or col not in clean_reference.columns:
            continue
        true_levels = clean_reference[col].nunique()
        if train[col].nunique() > true_levels:
            return {"present": True, "column": col, "true_levels": int(true_levels),
                    "raw_levels": int(train[col].nunique())}
    return {"present": False}


def run_all(train, later, holdout):
    return {
        "leaking_column": check_leaking_column(train, holdout),
        "unit_mixup": check_unit_mixup(train),
        "sentinel_value": check_sentinel(train),
        "duplicate_rows": check_duplicates(train),
        "stuck_sensor": check_stuck_sensor(train),
        "target_typos": check_target_typos(train),
        "spelling_variants": check_spelling_variants(train, later),
    }
