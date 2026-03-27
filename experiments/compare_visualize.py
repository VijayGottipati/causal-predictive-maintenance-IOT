"""
Compare three stacks on the same held-out data:
  1) optimized  — models/model.pkl + models/threshold.json (full project)
  2) ablation   — experiments/ablation_no_csos_cci_caq_iwl_pmw/artifacts/
  3) vanilla    — experiments/baseline_no_optimizations/artifacts/

Metrics: accuracy, precision, recall, F1, latency (predict_proba), model file size,
         noise robustness (F1 ratio with Gaussian noise on scaled features).

Run from project root after:
  python experiments/ablation_no_csos_cci_caq_iwl_pmw/run_experiment.py
  python experiments/baseline_no_optimizations/run_experiment.py

  python experiments/compare_visualize.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "experiments" / "results"
FIGURES = RESULTS / "figures"

NOISE_STD = 0.08
LATENCY_REPEATS = 30
WARMUP = 5


def _load_threshold() -> float:
    p = ROOT / "models" / "threshold.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return float(json.load(f).get("threshold", 0.5))
    return 0.5


def _predict_labels(model, X: np.ndarray, threshold: float) -> np.ndarray:
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)


def _latency_ms(model, X: np.ndarray) -> dict[str, float]:
    for _ in range(WARMUP):
        model.predict_proba(X)
    times: list[float] = []
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        model.predict_proba(X)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_std_ms": float(arr.std()),
        "latency_p99_ms": float(np.percentile(arr, 99)),
    }


def _noise_robustness(model, X: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    rng = np.random.default_rng(42)
    noise = rng.normal(0, NOISE_STD, size=X.shape)
    Xn = X + noise
    pred_clean = _predict_labels(model, X, threshold)
    pred_noisy = _predict_labels(model, Xn, threshold)
    f1_c = float(f1_score(y, pred_clean, average="weighted", zero_division=0))
    f1_n = float(f1_score(y, pred_noisy, average="weighted", zero_division=0))
    ratio = f1_n / f1_c if f1_c > 0 else 0.0
    return {
        "f1_clean_weighted": f1_c,
        "f1_noisy_weighted": f1_n,
        "noise_robustness_ratio": float(ratio),
    }


def _evaluate_variant(
    name: str,
    model_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if not model_path.exists():
        return {"variant": name, "error": f"missing model: {model_path}"}

    model = joblib.load(model_path)
    y_pred = _predict_labels(model, X_test, threshold)

    row: dict[str, Any] = {
        "variant": name,
        "threshold": threshold,
        "model_path": str(model_path.relative_to(ROOT)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "model_size_bytes": os.path.getsize(model_path),
    }
    row.update(_latency_ms(model, X_test))
    row.update(_noise_robustness(model, X_test, y_test, threshold))
    return row


def _plot(rows: list[dict[str, Any]]) -> None:
    ok = [r for r in rows if "error" not in r]
    if len(ok) < 2:
        print("Not enough variants to plot.")
        return

    names = [r["variant"] for r in ok]
    palette = {"optimized_full_stack": "#27ae60", "ablation_no_csos_cci_caq_iwl_pmw": "#3498db"}
    colors = [palette.get(n, "#e74c3c") for n in names]
    metrics = ["f1_weighted", "recall", "precision", "noise_robustness_ratio"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ax, m in zip(axes, metrics):
        vals = [r[m] for r in ok]
        ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(m.replace("_", " ").title())
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Variant comparison (same X_test, scaled features)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIGURES / "comparison_quality.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out}")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ok))
    w = 0.35
    ax2.bar(x - w / 2, [r["latency_mean_ms"] for r in ok], w, label="predict_proba mean (ms)", color="#e67e22")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=12)
    ax2.set_ylabel("Latency (ms)", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")
    ax2b = ax2.twinx()
    ax2b.bar(x + w / 2, [r["model_size_bytes"] / 1e6 for r in ok], w, label="model .pkl (MB)", color="#9b59b6", alpha=0.85)
    ax2b.set_ylabel("Model file size (MB)", color="#9b59b6")
    ax2b.tick_params(axis="y", labelcolor="#9b59b6")
    ax2.set_title("Inference latency vs serialized model size")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")
    fig2.tight_layout()
    out2 = FIGURES / "comparison_latency_size.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved figure -> {out2}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    X_test = np.load(DATA / "X_test.npy")
    y_test = np.load(DATA / "y_test.npy")

    rows = [
        _evaluate_variant(
            "optimized_full_stack",
            ROOT / "models" / "model.pkl",
            X_test,
            y_test,
            _load_threshold(),
        ),
        _evaluate_variant(
            "ablation_no_csos_cci_caq_iwl_pmw",
            ROOT / "experiments" / "ablation_no_csos_cci_caq_iwl_pmw" / "artifacts" / "model.pkl",
            X_test,
            y_test,
            0.5,
        ),
        _evaluate_variant(
            "baseline_no_optimizations",
            ROOT / "experiments" / "baseline_no_optimizations" / "artifacts" / "model.pkl",
            X_test,
            y_test,
            0.5,
        ),
    ]

    payload = {
        "noise_std_on_scaled_features": NOISE_STD,
        "latency_repeats": LATENCY_REPEATS,
        "variants": rows,
    }
    out_json = RESULTS / "comparison_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_json}")

    for r in rows:
        if "error" in r:
            print(f"[skip] {r['variant']}: {r['error']}")
        else:
            print(
                f"{r['variant']}: F1w={r['f1_weighted']:.4f} "
                f"lat_mean={r['latency_mean_ms']:.3f}ms "
                f"noise_ratio={r['noise_robustness_ratio']:.4f}"
            )

    _plot(rows)


if __name__ == "__main__":
    main()
