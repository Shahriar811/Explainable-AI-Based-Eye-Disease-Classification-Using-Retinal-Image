from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model identifier (bundle filename)")
    path: str = Field(..., description="Absolute path to the bundle on disk")
    architecture: Optional[str] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    image_size: Optional[int] = None


class PredictResponse(BaseModel):
    model_id: str
    predicted_index: int
    predicted_label: Optional[str]
    probabilities: List[float]
    top_k: List[Dict[str, Any]]


class PredictXaiResponse(PredictResponse):
    """Prediction + a Grad-CAM++ overlay PNG (base64-encoded)."""

    overlay_png_b64: str = Field(
        ...,
        description="PNG image showing the input with a Grad-CAM++ heatmap overlay, base64 encoded.",
    )

