"""
retrain.py
----------
Triggered automatically when drift is detected above threshold.
Person 4 (CI/CD) calls this via GitHub Actions when drift_score >= 0.3.

Usage:
    python src/retrain.py --batch_id 3
    python src/retrain.py --auto   # reads latest drift_timeline.json
"""

import os
import json
import argparse
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data, preprocess, split_data, simulate_drift

MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME      = "telco-churn-retraining"
DATA_PATH            = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DRIFT_TIMELINE_PATH  = "artifacts/drift_timeline.json"
DRIFT_THRESHOLD      = 0.3


def should_retrain(batch_id: int = None) -> bool:
    """Check drift_timeline.json to decide if retraining is needed."""
    if not os.path.exists(DRIFT_TIMELINE_PATH):
        print("[retrain] No drift timeline found. Run drift_detector.py first.")
        return False

    with open(DRIFT_TIMELINE_PATH) as f:
        timeline = json.load(f)

    summaries = timeline.get("drift_summaries", [])

    if batch_id is not None:
        summary = next((s for s in summaries if s["batch_id"] == batch_id), None)
        if summary:
            decision = summary["retrain_needed"]
            print(f"[retrain] Batch {batch_id}: drift_score={summary['drift_score']} → retrain={decision}")
            return decision
        else:
            print(f"[retrain] No summary found for batch {batch_id}.")
            return False

    # Auto mode: retrain if any batch triggered it
    return any(s["retrain_needed"] for s in summaries)


def retrain_model(X_train, y_train, X_test, y_test, trigger_batch: int = -1):
    """Retrain the best model and register new version to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load previous model info
    with open("models/best_model_info.json") as f:
        best_info = json.load(f)

    print(f"\n[retrain] Previous best: {best_info['name']} (AUC={best_info['auc']:.4f})")
    print(f"[retrain] Retraining triggered by batch {trigger_batch}...")

    # Use best model type from registry
    if best_info["name"] == "random_forest":
        new_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    else:
        new_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

    with mlflow.start_run(run_name=f"retrain_after_batch_{trigger_batch}"):
        mlflow.log_param("trigger_batch",    trigger_batch)
        mlflow.log_param("retrain_reason",   "drift_detected")
        mlflow.log_param("drift_threshold",  DRIFT_THRESHOLD)
        mlflow.log_param("model_type",       best_info["name"])

        # Train
        new_model.fit(X_train, y_train)

        # Evaluate
        y_pred = new_model.predict(X_test)
        y_prob = new_model.predict_proba(X_test)[:, 1]

        new_metrics = {
            "retrain_accuracy": accuracy_score(y_test, y_pred),
            "retrain_f1":       f1_score(y_test, y_pred),
            "retrain_auc":      roc_auc_score(y_test, y_prob),
        }
        mlflow.log_metrics(new_metrics)

        print(f"[retrain] New model -> Accuracy: {new_metrics['retrain_accuracy']:.4f} "
              f"| F1: {new_metrics['retrain_f1']:.4f} "
              f"| AUC: {new_metrics['retrain_auc']:.4f}")

        # Register new version
        mlflow.sklearn.log_model(new_model, artifact_path="model")
        run_id  = mlflow.active_run().info.run_id
        mv      = mlflow.register_model(f"runs:/{run_id}/model", best_info["registry_name"])
        print(f"[retrain] Registered new version: {mv.version}")

        # Overwrite best model artifacts
        joblib.dump(new_model, "models/best_model.pkl")
        best_info.update({
            "run_id":  run_id,
            "auc":     new_metrics["retrain_auc"],
            "version": mv.version,
        })
        with open("models/best_model_info.json", "w") as f:
            json.dump(best_info, f, indent=2)

        return new_metrics


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_id", type=int, default=None,
                        help="Check a specific batch for drift")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-check all batches in drift_timeline.json")
    parser.add_argument("--force", action="store_true",
                        help="Force retrain regardless of drift score")
    args = parser.parse_args()

    # Load data
    df = load_data(DATA_PATH)
    X, y, feature_names, scaler = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    batch_id = args.batch_id if args.batch_id is not None else -1

    if args.force or should_retrain(args.batch_id):
        metrics = retrain_model(X_train, y_train, X_test, y_test, trigger_batch=batch_id)
        print("\n✅ Retraining complete.")
        # Write exit code 0 for CI/CD (Person 4 reads this)
        with open("artifacts/retrain_result.json", "w") as f:
            json.dump({"status": "retrained", "metrics": metrics, "batch_id": batch_id}, f, indent=2)
    else:
        print("\n✅ No retraining needed. Model is stable.")
        with open("artifacts/retrain_result.json", "w") as f:
            json.dump({"status": "skipped", "reason": "drift_below_threshold"}, f, indent=2)
