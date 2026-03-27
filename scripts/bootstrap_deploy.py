"""
Prepare data + baseline model when artifacts are missing (Render build, fresh clone, CI).

Steps: download AI4I CSV (if needed) -> preprocess -> train baseline RF.
Run from repository root. Idempotent if model + arrays already exist (use --force to rebuild).

Usage:
  python scripts/bootstrap_deploy.py
  python scripts/bootstrap_deploy.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _have_artifacts() -> bool:
    return (
        (ROOT / "data" / "X_train.npy").exists()
        and (ROOT / "models" / "model.pkl").exists()
        and (ROOT / "data" / "scaler.pkl").exists()
    )


def _ensure_default_threshold() -> None:
    p = ROOT / "models" / "threshold.json"
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"threshold": 0.35}, f, indent=2)
    print(f"Wrote default {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap data and model for deploy/CI")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and retrain even if artifacts exist",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    if not args.force and _have_artifacts():
        print("Bootstrap skipped: data/X_train.npy and models/model.pkl already present.")
        _ensure_default_threshold()
        return

    py = sys.executable

    print("=== Bootstrap: download dataset ===")
    r = subprocess.run([py, str(ROOT / "scripts" / "download_data.py")], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

    print("=== Bootstrap: preprocess ===")
    r = subprocess.run([py, str(ROOT / "scripts" / "preprocess.py")], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

    print("=== Bootstrap: train baseline model ===")
    r = subprocess.run([py, str(ROOT / "scripts" / "train.py")], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)

    _ensure_default_threshold()
    print("=== Bootstrap complete ===")


if __name__ == "__main__":
    main()
