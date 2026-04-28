"""
train.py
--------
Trains multiple models on Telco Churn, tracks every experiment in MLflow.
Compares: Logistic Regression, Random Forest, XGBoost (if installed).
Registers the best model to MLflow Model Registry.
"""

import os
import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)
import joblib

# Project imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_data, preprocess, split_data

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "telco-churn-drift-aware"
MODEL_REGISTRY_NAME = "telco-churn-best-model"
DATA_PATH           = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
ARTIFACTS_DIR       = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ── Model zoo ─────────────────────────────────────────────────────────────────
def get_models():
    return {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"max_iter": 1000, "solver": "lbfgs", "C": 1.0},
        },
        "random_forest": {
            "model": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
            "params": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2},
        },
        "gradient_boosting": {
            "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
            "params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        },
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_score":  f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


# ── Artifacts ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    path = f"{ARTIFACTS_DIR}/{model_name}_confusion_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names: list, model_name: str) -> str | None:
    if not hasattr(model, "feature_importances_"):
        return None
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:15]   # top 15
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(indices)), importance[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    path = f"{ARTIFACTS_DIR}/{model_name}_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ── Core training loop ────────────────────────────────────────────────────────
def train_and_track(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    feature_names: list,
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models       = get_models()
    results      = {}
    best_run_id  = None
    best_auc     = -1

    print(f"\n{'='*60}")
    print(f"MLflow Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"Experiment          : {EXPERIMENT_NAME}")
    print(f"{'='*60}\n")

    for name, config in models.items():
        print(f"▶ Training {name}...")

        with mlflow.start_run(run_name=name) as run:
            # ── Log hyperparameters
            mlflow.log_params(config["params"])
            mlflow.log_param("model_type", name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("feature_count", len(feature_names))

            # ── Train
            model = config["model"]
            model.fit(X_train, y_train)

            # ── Evaluate on val + test
            for split_name, X_s, y_s in [
                ("val",  X_val,  y_val),
                ("test", X_test, y_test),
            ]:
                y_pred = model.predict(X_s)
                y_prob = model.predict_proba(X_s)[:, 1] if hasattr(model, "predict_proba") else None
                m = compute_metrics(y_s, y_pred, y_prob)

                # Log with split prefix
                mlflow.log_metrics({f"{split_name}_{k}": v for k, v in m.items()})

                if split_name == "test":
                    print(f"  Test  -> Accuracy: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | AUC: {m.get('roc_auc', 'N/A')}")

            # ── Artifacts
            cm_path = plot_confusion_matrix(y_test, model.predict(X_test), name)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            fi_path = plot_feature_importance(model, feature_names, name)
            if fi_path:
                mlflow.log_artifact(fi_path, artifact_path="plots")

            # ── Log classification report as text
            report = classification_report(y_test, model.predict(X_test),
                                           target_names=["No Churn", "Churn"])
            report_path = f"{ARTIFACTS_DIR}/{name}_classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path, artifact_path="reports")

            # ── Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None,   # register only the best later
            )

            # ── Save locally too (for Person 2's FastAPI)
            joblib.dump(model, f"models/{name}.pkl")

            # Track best
            test_auc = mlflow.active_run()
            run_id   = run.info.run_id
            auc_val  = m.get("roc_auc", m["accuracy"])
            results[name] = {"run_id": run_id, "auc": auc_val, "metrics": m}

            if auc_val > best_auc:
                best_auc    = auc_val
                best_run_id = run_id
                best_model  = model
                best_name   = name

    # ── Register best model ────────────────────────────────────────────────
    print(f"\n🏆 Best model: {best_name} (AUC={best_auc:.4f})")
    print(f"   Run ID: {best_run_id}")

    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
    print(f"   Registered as '{MODEL_REGISTRY_NAME}' version {mv.version}")

    # Save best model path for other team members
    joblib.dump(best_model, "models/best_model.pkl")
    with open("models/best_model_info.json", "w") as f:
        json.dump({
            "name": best_name,
            "run_id": best_run_id,
            "auc": best_auc,
            "registry_name": MODEL_REGISTRY_NAME,
            "version": mv.version,
        }, f, indent=2)
    print("   Saved → models/best_model.pkl  +  models/best_model_info.json")

    return results, best_name, best_run_id


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load & preprocess
    df = load_data(DATA_PATH)
    X, y, feature_names, scaler = preprocess(df)

    # Save scaler for inference
    joblib.dump(scaler, "models/scaler.pkl")

    # 2. Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Train all models, track in MLflow
    results, best_name, best_run_id = train_and_track(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        feature_names,
    )

    print("\n✅ Training complete. Open MLflow UI: mlflow ui --port 5000")
