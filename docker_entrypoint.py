#!/usr/bin/env python3
"""Container entrypoint: bootstrap artifacts if missing, then exec CMD (uvicorn)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    os.chdir(ROOT)
    model = ROOT / "models" / "model.pkl"
    data = ROOT / "data" / "X_train.npy"
    if not model.exists() or not data.exists():
        print("=== Missing artifacts; running bootstrap_deploy ===", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "bootstrap_deploy.py")],
            check=True,
            cwd=ROOT,
        )
    if len(sys.argv) < 2:
        raise SystemExit("no command after entrypoint")
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
