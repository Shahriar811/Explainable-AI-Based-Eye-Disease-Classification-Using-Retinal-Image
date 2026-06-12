# Model Serving API (Edge-friendly)

This folder contains a standalone HTTP API for serving your trained models (the `*_bundle.pth` files).

## What you get

- FastAPI server with:
  - `GET /` — friendly browser UI (pick a model from a dropdown, drop a retinal
    image, see the prediction). No typing, no Swagger schemas to scroll past.
  - `GET /models` — JSON list of available models
  - `POST /predict` — run a prediction
  - `GET /docs` — interactive API reference (for developers)
- Loads the same `*_bundle.pth` format used by your Streamlit tester (`app.py`)
- Works on CPU by default (better for edge); can use GPU if available

## Quick start (Windows)

1. Create and activate a venv:
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
2. Install deps:
   - `pip install -r requirements.txt`
3. Run:
   - `.\run.ps1`

Then open:
- `http://127.0.0.1:8000/` — friendly web UI (recommended)
- `http://127.0.0.1:8000/docs` — interactive API reference

## Configuration

Environment variables (optional):

- `MODEL_DIR`: Directory containing `*_bundle.pth` files  
  Default: `..\CSE498R models`
- `API_HOST`: Default `127.0.0.1`
- `API_PORT`: Default `8000`

## Notes for edge devices

- Prefer exporting models to ONNX for faster CPU inference on edge.
- This scaffold focuses on serving your current `.pth` bundles first; you can add ONNX later.

