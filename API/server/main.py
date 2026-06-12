from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .config import MODEL_DIR
from .image_utils import preprocess_image, read_image_bytes
from .model_loader import LoadedBundle, load_bundle, predict_proba
from .schemas import ModelInfo, PredictResponse, PredictXaiResponse
from .web import INDEX_HTML
from .xai import compute_gradcam_overlay_png


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="CSE498 Eye Disease Detection",
    version="0.1.0",
    description="Upload a retinal image and get a prediction from the selected model.",
    swagger_ui_parameters={
        # Hide the "Schemas" section at the bottom of /docs.
        "defaultModelsExpandDepth": -1,
        "docExpansion": "list",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bundle_paths() -> List[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted(MODEL_DIR.glob("*_bundle.pth"))


@lru_cache(maxsize=16)
def _get_loaded_bundle(bundle_path_str: str):
    bundle_path = Path(bundle_path_str)
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path_str)
    return load_bundle(bundle_path=bundle_path, device=DEVICE)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.get("/models", response_model=List[ModelInfo])
def list_models():
    infos: List[ModelInfo] = []
    for p in _bundle_paths():
        try:
            loaded = _get_loaded_bundle(str(p))
            infos.append(
                ModelInfo(
                    id=p.name,
                    path=str(p),
                    architecture=loaded.meta.architecture,
                    num_classes=len(loaded.meta.class_names),
                    class_names=loaded.meta.class_names,
                    image_size=loaded.meta.image_size,
                )
            )
        except Exception:
            infos.append(ModelInfo(id=p.name, path=str(p)))
    return infos


async def _run_inference(
    model_id: str, file: UploadFile, top_k: int
) -> Tuple[LoadedBundle, torch.Tensor, np.ndarray, int, List[Dict], Optional[str]]:
    """Shared logic for /predict and /predict_xai.

    Returns (loaded_bundle, preprocessed_tensor, probabilities, pred_idx,
    top_k_records, predicted_label).
    """
    candidates = {p.name: p for p in _bundle_paths()}
    bundle_path = candidates.get(model_id)
    if bundle_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id. Available: {sorted(candidates.keys())}",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty upload.")

    loaded = _get_loaded_bundle(str(bundle_path))
    pil_img = read_image_bytes(image_bytes)
    x = preprocess_image(
        pil_img=pil_img,
        image_size=loaded.meta.image_size,
        mean=loaded.meta.mean,
        std=loaded.meta.std,
    ).to(DEVICE)

    probs = predict_proba(loaded.model, x)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))

    k = max(1, min(int(top_k), len(probs)))
    top_idx = list(np.argsort(-probs)[:k])
    top_data: List[Dict] = [
        {
            "index": int(i),
            "label": loaded.meta.class_names[int(i)] if loaded.meta.class_names else None,
            "prob": float(probs[int(i)]),
        }
        for i in top_idx
    ]

    predicted_label: Optional[str] = None
    if loaded.meta.class_names and 0 <= pred_idx < len(loaded.meta.class_names):
        predicted_label = loaded.meta.class_names[pred_idx]

    return loaded, x, probs, pred_idx, top_data, predicted_label


@app.post("/predict", response_model=PredictResponse)
async def predict(
    model_id: str,
    file: UploadFile = File(...),
    top_k: int = 5,
):
    _loaded, _x, probs, pred_idx, top_data, predicted_label = await _run_inference(
        model_id, file, top_k
    )
    return PredictResponse(
        model_id=model_id,
        predicted_index=pred_idx,
        predicted_label=predicted_label,
        probabilities=[float(p) for p in probs.tolist()],
        top_k=top_data,
    )


@app.post("/predict_xai", response_model=PredictXaiResponse)
async def predict_xai(
    model_id: str,
    file: UploadFile = File(...),
    top_k: int = 5,
):
    """Same as /predict, plus a Grad-CAM++ overlay PNG (base64-encoded).

    Slightly slower than /predict because it runs an additional backward pass
    through the model to compute attention.
    """
    loaded, x, probs, pred_idx, top_data, predicted_label = await _run_inference(
        model_id, file, top_k
    )

    try:
        overlay_png = compute_gradcam_overlay_png(
            model=loaded.model,
            x_1chw=x,
            pred_idx=pred_idx,
            mean=loaded.meta.mean,
            std=loaded.meta.std,
        )
    except Exception as ex:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM++ failed for this model/image: {ex}",
        )

    overlay_b64 = base64.b64encode(overlay_png).decode("ascii")
    return PredictXaiResponse(
        model_id=model_id,
        predicted_index=pred_idx,
        predicted_label=predicted_label,
        probabilities=[float(p) for p in probs.tolist()],
        top_k=top_data,
        overlay_png_b64=overlay_b64,
    )

