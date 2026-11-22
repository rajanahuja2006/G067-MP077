"""FastAPI service for the MP077 pharmacy daily-sales model."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

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


class PredictionRequest(BaseModel):
    """The exact eight-feature request contract defined in PROBLEM.md."""

    model_config = ConfigDict(extra="forbid")

    flu_cases_reported: int
    temp_c: float
    nearby_clinics: int = Field(le=25)
    discount_pct: float = Field(ge=0)
    stock_lines: int
    distance_to_hospital_km: float
    shop_type: str
    locality: str


class PredictionResponse(BaseModel):
    prediction: float


class JsonLineFormatter(logging.Formatter):
    """Format each prediction log as one JSON object on stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        if hasattr(record, "prediction"):
            payload["prediction"] = record.prediction
        return json.dumps(payload, separators=(",", ":"), allow_nan=False)


class DynamicStdoutHandler(logging.StreamHandler):
    """Resolve stdout at emit time so tests and process supervisors capture it."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def _configure_prediction_logger() -> logging.Logger:
    logger = logging.getLogger("mp077.predictions")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = DynamicStdoutHandler(sys.stdout)
        handler.setFormatter(JsonLineFormatter())
        logger.addHandler(handler)
    return logger


prediction_logger = _configure_prediction_logger()


def _load_model() -> Any:
    configured_path = os.getenv("MODEL_PATH")
    model_path = (
        Path(configured_path)
        if configured_path
        else Path(__file__).resolve().parents[1] / "models" / "model.joblib"
    )
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Champion model not found at {model_path}. "
            "Ensure models/model.joblib is included in the repository and container."
        )
    model = joblib.load(model_path)
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features != FEATURES:
        raise ValueError(
            "Champion model feature order does not match the API contract: "
            f"expected {FEATURES}, got {model_features}"
        )
    return model


# Load once when the service starts; requests reuse the same in-memory model.
model = _load_model()

app = FastAPI(
    title="MP077 Pharmacy Daily Sales API",
    version="1.0.0",
    description="Predict daily medicine units sold for a pharmacy.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health endpoint used by Kubernetes readiness and liveness probes."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict units sold from the eight validated pharmacy features."""
    values = request.model_dump()
    features = pd.DataFrame([values], columns=FEATURES)
    prediction = float(model.predict(features)[0])

    # Deliberately log no request fields, shop details, or other customer data.
    prediction_logger.info(
        "prediction_generated",
        extra={"prediction": prediction},
    )
    return PredictionResponse(prediction=prediction)
