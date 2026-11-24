# MP077 - Pharmacy Daily Sales (Nagpur)

Group: **G067**  |  Problem: **MP077**  |  Faculty: Dhananjay Kumar  |  Batch 16

## Team Members & Contributions

| Enrolment | Name | GitHub | Assigned Stage & Contribution | Status |
|---|---|---|---|---|
| S24CSEU1218 | Rajan Ahuja | [rajanahuja2006](https://github.com/rajanahuja2006) | **Stage 1:** Repository setup, DVC tracking (`train.csv`, `later.csv`), data quality checks (`src/mp077/data_checks.py`), unit tests, and `reports/findings.yaml` | Completed (20 Nov 2026) |
| S24CSEU1213 | Kanchan Kumari | [kanchan3145](https://github.com/kanchan3145) | Stage 1 Review & **Stage 2:** Data Cleaning / Preprocessing | Completed (21 Nov 2026) |
| S24CSEU1220 | Aadya Gupta | [aadya197](https://github.com/aadya197) | Stage 1 Review & **Stage 2:** Model Training, MLflow Tracking (`mp077-stage2`, `@champion`) & Holdout Prediction | Completed (22 Nov 2026) |
| S24CSEU1234 | Bhumik Kundu | [Bhumikkundu001](https://github.com/Bhumikkundu001) | Stage 1 Review & Stage 3 — API, Docker, CI and Kubernetes | Completed (24 Nov 2026) |
| S24CSEU1223 | Animesh Kumar Yadav | [Animesh160604](https://github.com/Animesh160604) | Stage 1 Review & Stage 4 — Monitoring, retraining and final predictions | In Progress |

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
If no DVC remote is configured, restore the original assignment datasets
to `data/raw/train.csv` and `data/raw/later.csv`, verify their SHA256
fingerprints, and populate the local DVC cache using `dvc commit`.

Restore the DVC-tracked datasets first (a configured DVC remote is required):
```bash
dvc pull
PYTHONPATH=src python3 -m mp077.run_stage1
```

---

## Stage 2: Data Cleaning, Model Training & Experiment Tracking (Completed)
* **Completed by:** Aadya Gupta (`aadya197`)
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
**Important:** These training commands are retained for historical reproducibility.
Do not rerun Stage 2 training against the final MLflow database, as it already
contains V1 and V2 with V2 assigned the `champion` alias.

```bash
# Train models and track in MLflow
PYTHONPATH=src python3 -m mp077.run_stage2 train --train data/raw/train.csv --later data/raw/later.csv --data-sha 8d00fd3a8a17d19bbf8db246e5047df5d769eb6af22f54fee0012748ffc034a3

# Generate holdout predictions
PYTHONPATH=src python3 -m mp077.run_stage2 predict --holdout data/raw/holdout_features.csv --out predictions.csv
```
---

## Stage 3: API Deployment & CI (Completed)

* **Completed by:** Bhumik Kundu (`Bhumikkundu001`)
* **FastAPI Application (`app/main.py`):**
  * Implemented `GET /health` endpoint for application health checks.
  * Implemented `POST /predict` endpoint to generate predictions using `models/model.joblib`.
  * Added Pydantic request validation with the required constraints for `nearby_clinics` and `discount_pct`.
* **JSON Logging:**
  * Implemented structured JSON logging to stdout.
  * Configured one JSON log entry per prediction without logging personal details.
* **Docker Configuration:**
  * Created a `Dockerfile` using `python:3.12-slim`.
  * Configured Uvicorn to serve the application on port `8000`.
  * Added `.dockerignore` to exclude unnecessary files from the Docker build context.
* **Kubernetes Configuration:**
  * Created `k8s/deployment.yaml` for Deployment `mp077-api` with 4 replicas.
  * Initially configured container image `mp077-api:1.0` for V1. Stage 4 updates the deployment to `mp077-api:2.0` for Champion V2. Port `8000` and `/health` probes remain unchanged.
  * Created `k8s/service.yaml` for Service `mp077-svc`, exposing port `80` and forwarding to port `8000`.
  * Live cluster deployment was not required by the Stage 3 handbook.
* **Continuous Integration (GitHub Actions):**
  * Created `.github/workflows/ci.yml` to automatically install dependencies and run pytest on pushes and pull requests.
  * Added `httpx==0.28.1` to `requirements.txt` for FastAPI TestClient support.
  * Successfully verified the workflow on GitHub Actions.
* **Testing Results:**
  * Local API and project tests: `10 passed`.
  * GitHub Actions: Successful workflow run (`639b0e3`).

### How to Run Stage 3

```bash
# Install project dependencies
python -m pip install -r requirements.txt

# Start the FastAPI application
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests
python -m pytest -q

# Build Docker image
docker build -t mp077-api:1.0 .
```

---

## Stage 4: Monitoring, Retraining & Final Predictions (Completed)

* **Completed by:** Kanchan Kumari (`kanchan3145`)
* **Drift Monitoring (`src/mp077/monitor_stage4.py`):**
  * Calculated Population Stability Index (PSI) for all six numerical features using 10 equal-width bins.
  * Compared the cleaned training dataset (1,000 records) against the later dataset (300 records).
  * Identified `nearby_clinics` as the most drifted feature with a PSI of `3.892839`.
  * Evaluated V1 model performance across all three localities.
  * Identified `New Town` as the worst served locality with an MAE of `238.489265`.

* **Model Retraining (`src/mp077/run_stage4.py`):**
  * Chronologically split the later dataset into 70% training (210 records) and 30% evaluation (90 records).
  * Combined the cleaned original training data with the first 70% of later data, resulting in 1,210 training records.
  * Retrained the LinearRegression pipeline without including the evaluation records.
  * Evaluated both V1 and V2 on the same 90-record check dataset.

* **MLflow Tracking & Champion Promotion (`mlflow.db`):**
  * Created the `mp077-stage4` experiment to track retraining metrics, parameters and dataset fingerprints.
  * Registered the retrained model as `mp077-model` version `2`.
  * V1 check MAE: `153.780285`.
  * V2 check MAE: `90.472110`.
  * Achieved an MAE reduction of `41.17%`, exceeding the required 10% improvement threshold.
  * Promoted V2 as `@champion` while preserving V1 in the model registry.
  * Verified both registered model versions and all 8 MLflow runs.
  * Migrated MLflow artifact paths to repository-relative paths for improved portability.

* **Final Holdout Predictions (`src/mp077/predict_holdout.py`):**
  * Generated predictions for all 250 holdout records using Champion V2.
  * Saved final predictions to `predictions/holdout_predictions.csv`.
  * Verified the required `record_id,prediction` schema, unique record IDs and valid prediction values.

* **Monitoring Report (`reports/findings.yaml`):**
  * Updated PSI values for all six numerical features.
  * Recorded locality-wise MAE, most drifted feature and worst served locality.
  * Recorded V1/V2 evaluation metrics and the final promotion decision.
  * Preserved original data quality findings and Stage 2 tracking information.
  * Independently verified consistency between YAML findings and MLflow metrics.

* **Champion Export & Deployment:**
  * Exported Champion V2 to `models/model.joblib`.
  * Updated model metadata and reference predictions.
  * Built and verified Docker image `mp077-api:2.0`.
  * Successfully performed a Kubernetes rolling update to Champion V2.
  * Verified model integrity across all 4 running replicas.
  * Preserved the original V1 Docker image.

* **Testing Results:**
  * Local API and project tests: `10 passed`.
  * Docker V2 prediction verification: `5/5 passed`.
  * Kubernetes replicas: `4/4 ready`, with zero restarts.
  * Kubernetes Service prediction requests: `40/40 successful`.
  * All four pods received prediction requests.
  * All service predictions matched Champion V2 reference values.
  * Independent findings and MLflow database integrity checks: `PASS`.

### How to Run Stage 4

Restore the original DVC-tracked datasets before running monitoring.

```bash
# Install project dependencies
python -m pip install -r requirements.txt

# Run drift and performance monitoring
PYTHONPATH=src python3 -m mp077.monitor_stage4

# Verify the exported champion model
PYTHONPATH=src python3 -m mp077.export_model check

# Run automated tests
python -m pytest -q

# Build the Champion V2 Docker image
docker build -t mp077-api:2.0 .

# Deploy Champion V2 to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verify deployment
kubectl rollout status deployment/mp077-api
kubectl get pods
```

**Note:** The final MLflow database already contains V1 and V2, with V2 assigned the `champion` alias. Do not rerun training or registration scripts against the completed database.

**Limitations:** PSI uses 10 equal-width bins with an epsilon of 0.0001. Holdout target values are unavailable for local accuracy evaluation. Kubernetes testing verifies sequential functional requests rather than concurrent stress performance.