# MP077 - Pharmacy Daily Sales (Nagpur)

Group: **G067**  |  Problem: **MP077**  |  Faculty: Dhananjay Kumar  |  Batch 16

## Team Members & Contributions

| Enrolment | Name | GitHub | Assigned Stage & Contribution | Status |
|---|---|---|---|---|
| S24CSEU1218 | Rajan Ahuja | [rajanahuja2006](https://github.com/rajanahuja2006) | **Stage 1:** Repository setup, DVC tracking (`train.csv`, `later.csv`), data quality checks (`src/mp077/data_checks.py`), unit tests, and `reports/findings.yaml` | Completed (20 Nov 2026) |
| S24CSEU1213 | Kanchan Kumari | [kanchan3145](https://github.com/kanchan3145) | Stage 1 Review & **Stage 2:** Data Cleaning / Preprocessing | Completed (21 Nov 2026) |
| S24CSEU1220 | Aadya Gupta | [aadya197](https://github.com/aadya197) | Stage 1 Review & **Stage 2:** Model Training, MLflow Tracking (`mp077-stage2`, `@champion`) & Holdout Prediction | Completed (22 Nov 2026) |
| S24CSEU1234 | Bhumik Kundu | [Bhumikkundu001](https://github.com/Bhumikkundu001) | Stage 1 Review & Stage 3 (Model Training / MLflow) | In Progress |
| S24CSEU1223 | Animesh Kumar Yadav | [Animesh160604](https://github.com/Animesh160604) | Stage 1 Review & Stage 4 (API / CI) | In Progress |

---

## Stage 1: Data Versioning & Quality Checks (Completed)
* **Completed by:** Rajan Ahuja (`rajanahuja2006`)
* **DVC Tracking:** Configured DVC to track large raw datasets (`data/raw/train.csv.dvc` and `data/raw/later.csv.dvc`) while keeping raw CSVs out of Git.
* **Automated Data Checks (`src/mp077/data_checks.py`):** Implemented detection across all 7 data quality checks and recorded results in `reports/findings.yaml`:
  * **Unit Mixup (`discount_pct`):** Detected factor `0.01` scaling discrepancy in `"Old Town"` (`observed_ratio: 0.0107`).
  * **Duplicate Rows:** Identified `25` exact duplicate records.
  * **Spelling Variants (`shop_type`):** Flagged `19` raw categorical variants mapping to `3` true levels.
  * **Clean Checks:** Verified absence of target leakage, sentinel values, stuck sensors, and target typos.

### How to Run Stage 1
Restore the DVC-tracked datasets first (a configured DVC remote is required):
```bash
dvc pull
PYTHONPATH=src python3 -m mp077.run_stage1
```

---

## Stage 2: Data Cleaning, Model Training & Experiment Tracking (Completed)
* **Completed by:** Aadya Gupta (`aadya197`) & Kanchan Kumari (`kanchan3145`)
* **Cleaning Pipeline (`src/mp077/run_stage2.py`):**
  * Handled duplicate rows (`25` extra records removed).
  * Standardized `19` raw `shop_type` spelling variants to `3` true levels (`chain`, `hospital_store`, `independent`).
  * Fixed unit mixup in `discount_pct` for `"Old Town"` (`0.01` scale factor).
* **Experiment Tracking & Model Registry (`mlflow.db`):**
  * Configured MLflow SQLite backend (`sqlite:///mlflow.db`) under experiment `mp077-stage2`.
  * Tracked models: `baseline-mean` (Dummy), `linear-baseline` (LinearRegression), `tree-depth-{2,4,6,12}` (DecisionTreeRegressor), and 5-fold CV hyperparameter-tuned `rf-tuned` (RandomForestRegressor).
  * Champion Model: `linear-baseline` achieved lowest test MAE of `20.683` (beating mean baseline of `92.896` by `72.213`).
  * Registered champion model as `mp077-model` version `1` with alias `@champion`.
  * Generated holdout predictions on `data/raw/holdout_features.csv` and saved to `predictions/holdout_predictions.csv` and `predictions.csv`.

### How to Run Stage 2
```bash
# Train models and track in MLflow
PYTHONPATH=src python3 -m mp077.run_stage2 train --train data/raw/train.csv --later data/raw/later.csv --data-sha 8d00fd3a8a17d19bbf8db246e5047df5d769eb6af22f54fee0012748ffc034a3

# Generate holdout predictions
PYTHONPATH=src python3 -m mp077.run_stage2 predict --holdout data/raw/holdout_features.csv --out predictions.csv
```
