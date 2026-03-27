# Models directory

Trained artifacts (`*.pkl`, `*.onnx`) are **not committed** (see root `.gitignore`).

**On CI, Render, or Docker**, run `python scripts/bootstrap_deploy.py` to download data, preprocess, and train a baseline `model.pkl`, or train locally and deploy the file via your pipeline.

Checked-in JSON files here (`threshold.json`, `mlp_config.json`, etc.) are small config defaults.
