"""
Optional MLflow tracking. Set MLFLOW_TRACKING_URI (and optionally MLFLOW_EXPERIMENT_NAME)
to log training runs. Safe no-op when unset or mlflow not installed.
"""

from __future__ import annotations

import os
from typing import Any


def tracking_enabled() -> bool:
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def log_training_run(
    *,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    experiment_name: str | None = None,
) -> None:
    if not tracking_enabled():
        return
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed; skip logging (pip install mlflow)")
        return

    exp = experiment_name or os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "predictive-maintenance"
    )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(exp)

    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        if metrics:
            clean = {k: float(v) for k, v in metrics.items() if v is not None}
            mlflow.log_metrics(clean)
