"""Render a static architecture diagram for the research paper."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")

PALETTE = {
    "data":  "#3498db",
    "ml":    "#9b59b6",
    "api":   "#16a085",
    "obs":   "#e67e22",
    "ci":    "#c0392b",
    "user":  "#34495e",
}


def box(x, y, w, h, text, color, text_color="white"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06",
        linewidth=1.5,
        edgecolor="black",
        facecolor=color,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        color=text_color, fontsize=10, fontweight="bold",
        wrap=True,
    )


def arrow(x1, y1, x2, y2, label="", color="#2c3e50", curve=0.0):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=15,
        color=color,
        lw=1.6,
        connectionstyle=f"arc3,rad={curve}",
    )
    ax.add_patch(a)
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label,
            fontsize=8, color=color, ha="center", style="italic",
        )


# ── Lanes (background) ───────────────────────────────────────────────────────
ax.text(7, 8.7, "Drift-Aware MLOps Pipeline with Adaptive Retraining",
        fontsize=15, fontweight="bold", ha="center")
ax.text(7, 8.35, "Telco Customer Churn  •  MLflow + Evidently + FastAPI + Prometheus + Grafana + GitHub Actions",
        fontsize=9, ha="center", color="#555")

# Person 1 — ML / MLflow lane
ax.add_patch(mpatches.Rectangle((0.2, 5.3), 4.4, 2.6, facecolor="#f4ecf7", edgecolor="none"))
ax.text(0.4, 7.7, "Person 1 — ML + MLflow", fontsize=9, color=PALETTE["ml"], fontweight="bold")
box(0.5, 6.7, 1.8, 0.7, "Telco Churn\nCSV (7,043×21)", PALETTE["data"])
box(2.6, 6.7, 1.9, 0.7, "data_loader.py\n(preprocess + drift sim)", PALETTE["ml"])
box(0.5, 5.5, 1.8, 0.9, "train.py\n(LR / RF / GB)", PALETTE["ml"])
box(2.6, 5.5, 1.9, 0.9, "MLflow Tracking\n+ Model Registry", PALETTE["ml"])

arrow(2.3, 7.05, 2.6, 7.05)
arrow(3.55, 6.7, 3.55, 6.4)
arrow(2.3, 5.95, 2.6, 5.95)
arrow(1.4, 6.7, 1.4, 6.4)

# Person 2 — FastAPI lane
ax.add_patch(mpatches.Rectangle((4.9, 5.3), 4.4, 2.6, facecolor="#e8f8f5", edgecolor="none"))
ax.text(5.1, 7.7, "Person 2 — FastAPI + Docker", fontsize=9, color=PALETTE["api"], fontweight="bold")
box(5.1, 6.7, 4.0, 0.7, "FastAPI service (api.py)", PALETTE["api"])
box(5.1, 5.5, 1.2, 0.9, "/predict", PALETTE["api"])
box(6.5, 5.5, 1.2, 0.9, "/drift-\nreport", PALETTE["api"])
box(7.9, 5.5, 1.2, 0.9, "/metrics\n/health", PALETTE["api"])

arrow(4.5, 7.05, 5.1, 7.05, label="best_model.pkl")
arrow(5.7, 6.7, 5.7, 6.4)
arrow(7.1, 6.7, 7.1, 6.4)
arrow(8.5, 6.7, 8.5, 6.4)

# Person 3 — Observability lane
ax.add_patch(mpatches.Rectangle((9.6, 5.3), 4.2, 2.6, facecolor="#fdf2e9", edgecolor="none"))
ax.text(9.8, 7.7, "Person 3 — Prometheus + Grafana", fontsize=9, color=PALETTE["obs"], fontweight="bold")
box(9.8, 6.7, 1.8, 0.7, "Prometheus\n(scrape 5s)", PALETTE["obs"])
box(11.8, 6.7, 1.8, 0.7, "Grafana\nDashboard", PALETTE["obs"])
box(9.8, 5.5, 3.8, 0.9,
    "Histogram: prediction_confidence  •  Gauge: drift_score\n"
    "Counter: predictions_total / retraining_events_total",
    PALETTE["obs"])
arrow(9.1, 7.05, 9.8, 7.05, label="GET /metrics")
arrow(11.6, 7.05, 11.8, 7.05)
arrow(10.7, 6.7, 10.7, 6.4)

# Person 4 — CI/CD lane
ax.add_patch(mpatches.Rectangle((0.2, 1.6), 13.6, 3.5, facecolor="#fdedec", edgecolor="none"))
ax.text(0.4, 4.85, "Person 4 — GitHub Actions CI/CD",
        fontsize=9, color=PALETTE["ci"], fontweight="bold")

box(1.0, 3.4, 2.4, 1.0, "drift_detector.py\n(Evidently:\n DataDriftPreset)", PALETTE["ci"])
box(4.0, 3.4, 2.4, 1.0, "Threshold check\n share_drifted ≥ 0.30", PALETTE["ci"])
box(7.0, 3.4, 2.6, 1.0, "GitHub repository_dispatch\n event = drift_detected", PALETTE["ci"])
box(10.2, 3.4, 3.0, 1.0, ".github/workflows/\nretrain.yml runs\nretrain.py --auto", PALETTE["ci"])

arrow(3.4, 3.9, 4.0, 3.9)
arrow(6.4, 3.9, 7.0, 3.9, label="≥ 0.30")
arrow(9.6, 3.9, 10.2, 3.9, label="POST /dispatches")
arrow(11.7, 3.4, 11.7, 2.6, label="register new\nmodel version")
arrow(11.7, 2.6, 4.0, 2.6, curve=0.15)
arrow(4.0, 2.6, 4.0, 5.5, curve=0.0)

box(3.0, 2.0, 2.4, 0.7, "MLflow Model Registry\n(versioned best model)", PALETTE["ml"])

# User
box(5.6, 0.4, 2.8, 0.8, "Production traffic\n(client / batch jobs)", PALETTE["user"])
arrow(7.0, 1.2, 7.0, 1.6, curve=0.0)
arrow(7.0, 5.5, 7.0, 5.0, curve=0.0)

# Legend
legend_elems = [
    mpatches.Patch(color=PALETTE["data"], label="Data"),
    mpatches.Patch(color=PALETTE["ml"],   label="ML / MLflow"),
    mpatches.Patch(color=PALETTE["api"],  label="API"),
    mpatches.Patch(color=PALETTE["obs"],  label="Observability"),
    mpatches.Patch(color=PALETTE["ci"],   label="CI/CD"),
    mpatches.Patch(color=PALETTE["user"], label="User"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=8, ncol=3, frameon=False)

plt.tight_layout()
out = "docs/architecture/architecture.png"
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print(f"saved -> {out}")
