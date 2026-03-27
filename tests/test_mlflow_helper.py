import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_mlflow_helper_no_tracking_uri_is_safe():
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    from scripts.mlflow_helper import log_training_run, tracking_enabled

    assert tracking_enabled() is False
    log_training_run(metrics={"f1_weighted": 0.99}, params={"k": "v"})
