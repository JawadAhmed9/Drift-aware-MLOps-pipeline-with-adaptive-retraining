"""Background traffic generator — keeps the API busy so Prometheus + Grafana
graphs have movement during demo / screenshots."""
import json
import random
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from data_loader import load_data, preprocess, split_data, simulate_drift  # noqa: E402
import pandas as pd  # noqa: E402

df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X, y, fn, _ = preprocess(df)
_, _, Xte, _, _, _ = split_data(X, y)


def predict_one():
    feat = dict(zip(fn, Xte[random.randrange(len(Xte))].tolist()))
    body = json.dumps({"features": feat}).encode()
    req = urllib.request.Request(
        "http://localhost:8000/predict",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req).read()


def drift_one(drifted: bool, batch_id: int):
    sample = Xte[: 100 + random.randrange(40)]
    if drifted:
        sample = simulate_drift(sample, "gradual", random.uniform(1.5, 2.8))
    batch = pd.DataFrame(sample, columns=fn).to_dict(orient="records")
    body = json.dumps({"batch_records": batch, "batch_id": batch_id}).encode()
    req = urllib.request.Request(
        "http://localhost:8000/drift-report",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req).read()


print("[traffic_gen] starting...")
counter = 0
while True:
    for _ in range(random.randrange(3, 9)):
        predict_one()
    counter += 1
    drift_one(drifted=(counter % 2 == 0), batch_id=counter)
    time.sleep(random.uniform(2.0, 4.0))
