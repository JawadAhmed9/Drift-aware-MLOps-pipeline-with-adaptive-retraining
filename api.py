from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import pandas as pd
import joblib
import json
import os
import sys
import time

from prometheus_client import (
    Counter, Gauge, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
)

# Project imports for drift detection
sys.path.insert(0, os.path.dirname(__file__))
from drift_detector import detect_drift
from data_loader import load_data, preprocess, split_data

app = FastAPI(title="Telco Churn API & Drift Detector", version="1.1")

# Globals for models and reference data
best_model = None
scaler = None
feature_names = None
reference_df = None

# ── Prometheus instrumentation (Person 3) ─────────────────────────────────────
REGISTRY = CollectorRegistry()

predictions_total = Counter(
    "mlops_predictions_total",
    "Total churn predictions made by the API",
    ["churn_label"],
    registry=REGISTRY,
)

prediction_confidence = Histogram(
    "mlops_prediction_confidence",
    "Distribution of churn-probability scores returned by /predict",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

drift_score_gauge = Gauge(
    "mlops_drift_score",
    "Latest dataset drift score (share of drifted features) from Evidently",
    registry=REGISTRY,
)

drift_features_gauge = Gauge(
    "mlops_drifted_features",
    "Latest count of drifted features",
    registry=REGISTRY,
)

drift_reports_total = Counter(
    "mlops_drift_reports_total",
    "Total drift reports generated",
    ["retrain_needed"],
    registry=REGISTRY,
)

retraining_events_total = Counter(
    "mlops_retraining_events_total",
    "Total retraining events observed (via best_model_info.json mtime change)",
    registry=REGISTRY,
)

last_retrain_timestamp = Gauge(
    "mlops_last_retrain_timestamp_seconds",
    "Unix timestamp of the most recent retraining event",
    registry=REGISTRY,
)

model_info_gauge = Gauge(
    "mlops_model_info",
    "Currently-loaded model metadata (always 1; values exposed via labels)",
    ["model_name", "version"],
    registry=REGISTRY,
)

# Track best_model_info.json mtime so we can detect retrain events
_BEST_INFO_PATH = "models/best_model_info.json"
_last_seen_mtime = 0.0


def _refresh_retrain_metric():
    """Detect retrain by watching best_model_info.json mtime."""
    global _last_seen_mtime
    if not os.path.exists(_BEST_INFO_PATH):
        return
    mtime = os.path.getmtime(_BEST_INFO_PATH)
    if mtime > _last_seen_mtime and _last_seen_mtime > 0:
        retraining_events_total.inc()
    if mtime > _last_seen_mtime:
        last_retrain_timestamp.set(mtime)
        _last_seen_mtime = mtime
        try:
            with open(_BEST_INFO_PATH) as f:
                info = json.load(f)
            model_info_gauge.clear()
            model_info_gauge.labels(
                model_name=info.get("name", "unknown"),
                version=str(info.get("version", "0")),
            ).set(1)
        except Exception:
            pass


@app.on_event("startup")
def load_assets():
    global best_model, scaler, feature_names, reference_df
    try:
        best_model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")

        data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if os.path.exists(data_path):
            df = load_data(data_path)
            X, y, f_names, _ = preprocess(df)
            feature_names = f_names
            X_train, _, _, _, _, _ = split_data(X, y)
            reference_df = pd.DataFrame(X_train, columns=feature_names)
        else:
            print(f"Warning: Data not found at {data_path}. Drift detection will fail.")

        # Initialise retrain metric from current model file
        if os.path.exists(_BEST_INFO_PATH):
            global _last_seen_mtime
            _last_seen_mtime = os.path.getmtime(_BEST_INFO_PATH)
            last_retrain_timestamp.set(_last_seen_mtime)
            with open(_BEST_INFO_PATH) as f:
                info = json.load(f)
            model_info_gauge.labels(
                model_name=info.get("name", "unknown"),
                version=str(info.get("version", "0")),
            ).set(1)
    except Exception as e:
        print(f"Startup error: {e}")


class ChurnRequest(BaseModel):
    features: dict


class DriftRequest(BaseModel):
    batch_records: list[dict]
    batch_id: int = 1


@app.post("/predict")
def predict_churn(request: ChurnRequest):
    if not best_model or not scaler or not feature_names:
        raise HTTPException(status_code=500, detail="Model or assets not loaded.")

    try:
        df = pd.DataFrame([request.features])
        df = df.reindex(columns=feature_names, fill_value=0)

        prediction = best_model.predict(df.values)
        probability = (
            best_model.predict_proba(df.values)[:, 1]
            if hasattr(best_model, "predict_proba")
            else None
        )

        label = int(prediction[0])
        predictions_total.labels(churn_label=str(label)).inc()
        if probability is not None:
            prediction_confidence.observe(float(probability[0]))

        return {
            "churn_prediction": label,
            "churn_probability": float(probability[0]) if probability is not None else None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/drift-report")
def generate_drift_report(request: DriftRequest):
    if reference_df is None:
        raise HTTPException(status_code=500, detail="Reference data not available.")

    try:
        prod_df = pd.DataFrame(request.batch_records)
        prod_df = prod_df.reindex(columns=feature_names, fill_value=0)

        summary = detect_drift(
            reference=reference_df,
            production=prod_df,
            batch_id=request.batch_id,
            log_to_mlflow=False,
        )

        drift_score_gauge.set(float(summary.get("drift_score", 0.0)))
        drift_features_gauge.set(int(summary.get("drifted_features", 0)))
        drift_reports_total.labels(
            retrain_needed=str(bool(summary.get("retrain_needed", False))).lower()
        ).inc()

        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def get_prometheus_metrics():
    _refresh_retrain_metric()
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": best_model is not None}
