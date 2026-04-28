"""
data_loader.py
--------------
Loads and preprocesses the Telco Customer Churn dataset.
Also provides drift simulation by injecting distribution shifts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


# ── Column config ────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL = "Churn"


def load_data(path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load raw CSV. Download from Kaggle if not present."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "Place it at: data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    df = pd.read_csv(path)
    print(f"[data_loader] Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def preprocess(df: pd.DataFrame):
    """
    Clean, encode, scale the raw dataframe.
    Returns: X (np.ndarray), y (np.ndarray), feature_names (list), scaler
    """
    df = df.copy()

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # TotalCharges comes as string with spaces → coerce to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # Encode categoricals
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # Scale numerical features
    scaler = StandardScaler()
    num_idx = [feature_cols.index(c) for c in NUMERICAL_COLS if c in feature_cols]
    X[:, num_idx] = scaler.fit_transform(X[:, num_idx])

    print(f"[data_loader] Preprocessed -> X: {X.shape}, y distribution: {np.bincount(y)}")
    return X, y, feature_cols, scaler


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split into train / val / test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )
    print(f"[data_loader] Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Drift Simulation ─────────────────────────────────────────────────────────

def simulate_drift(X: np.ndarray, drift_type: str = "gradual", strength: float = 2.0) -> np.ndarray:
    """
    Simulate data drift on production data for research comparison.

    drift_type:
        'gradual'  — slowly shifting mean over time (most realistic)
        'sudden'   — abrupt distribution shift
        'feature'  — only a subset of features drift
        'none'     — no drift (baseline)

    strength: how many std devs to shift (default 2.0)
    """
    X_drifted = X.copy().astype(float)
    rng = np.random.default_rng(seed=99)

    if drift_type == "none":
        print("[drift_sim] No drift applied.")
        return X_drifted

    elif drift_type == "sudden":
        # Shift ALL numerical features by `strength` std devs
        X_drifted += rng.normal(loc=strength, scale=0.5, size=X_drifted.shape)
        print(f"[drift_sim] Sudden drift applied (strength={strength}).")

    elif drift_type == "gradual":
        # Each row gets progressively more noise
        n = len(X_drifted)
        for i in range(n):
            scale = strength * (i / n)          # ramps from 0 -> strength
            X_drifted[i] += rng.normal(0, scale, size=X_drifted.shape[1])
        print(f"[drift_sim] Gradual drift applied (max strength={strength}).")

    elif drift_type == "feature":
        # Only first 3 features drift (e.g., tenure, charges change but not contract type)
        drifting_cols = [0, 1, 2]
        for col in drifting_cols:
            X_drifted[:, col] += rng.normal(loc=strength, scale=0.3, size=len(X_drifted))
        print(f"[drift_sim] Feature drift applied on cols {drifting_cols} (strength={strength}).")

    return X_drifted


def get_production_batches(X_test: np.ndarray, n_batches: int = 5, drift_start: int = 2):
    """
    Simulate production traffic arriving in batches.
    Batches 0..drift_start-1 → clean data
    Batches drift_start..n_batches-1 → drifted data

    Returns list of (batch_id, X_batch, is_drifted)
    """
    batch_size = len(X_test) // n_batches
    batches = []

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size if i < n_batches - 1 else len(X_test)
        X_batch = X_test[start:end]

        if i >= drift_start:
            X_batch = simulate_drift(X_batch, drift_type="gradual", strength=1.5 + i * 0.3)
            is_drifted = True
        else:
            is_drifted = False

        batches.append((i, X_batch, is_drifted))
        print(f"[batch] Batch {i}: {len(X_batch)} samples | drifted={is_drifted}")

    return batches
