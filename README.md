# Person 1: ML + MLflow — Telco Churn Drift-Aware Pipeline

## Your Folder Structure
```
mlops_pipeline/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   ← download this
├── src/
│   ├── data_loader.py      ← preprocessing + drift simulation
│   ├── train.py            ← model training + MLflow tracking
│   ├── drift_detector.py   ← Evidently drift detection
│   └── retrain.py          ← retraining trigger (used by Person 4)
├── models/                 ← saved .pkl files (used by Person 2)
├── artifacts/              ← plots, reports, drift timeline
└── requirements.txt
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place at: data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# 4. Start MLflow server (keep this running)
mlflow server --host 0.0.0.0 --port 5000
```

## Run Order

```bash
# Step 1 — Train all models, track in MLflow
python src/train.py

# Step 2 — Detect drift on simulated production batches
python src/drift_detector.py

# Step 3 — Retrain if drift detected (also triggered by Person 4 via CI/CD)
python src/retrain.py --auto
```

## What Gets Produced

| File | Used By |
|------|---------|
| `models/best_model.pkl` | Person 2 (FastAPI) |
| `models/scaler.pkl` | Person 2 (FastAPI) |
| `models/best_model_info.json` | Person 2, Person 4 |
| `artifacts/drift_timeline.json` | Person 3 (Prometheus) |
| `artifacts/drift_reports/*.html` | Research paper |
| `artifacts/accuracy_over_batches.png` | Research paper (key figure) |

## MLflow UI
Open: http://localhost:5000
- Experiment: `telco-churn-drift-aware`
- Models registry: `telco-churn-best-model`

## Drift Simulation Logic
- Batches 0–1: clean production data (matches training distribution)
- Batches 2–4: gradual drift injected (simulates real-world distribution shift)
- Drift threshold: 30% of features drifted → triggers retraining

## Key Research Output
`artifacts/accuracy_over_batches.png` — your main paper figure showing:
- Red line: accuracy degrading without retraining
- Green line: accuracy recovering with drift-aware retraining
