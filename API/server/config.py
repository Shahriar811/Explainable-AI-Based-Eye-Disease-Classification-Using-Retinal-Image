from __future__ import annotations

import os
from pathlib import Path


def _default_model_dir() -> Path:
    # Default points to the folder that already exists in your repo root.
    # Use a relative path so it works on other machines too.
    return (Path(__file__).resolve().parents[2] / "CSE498R models").resolve()


MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(_default_model_dir()))).expanduser().resolve()
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))

