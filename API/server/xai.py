"""
Grad-CAM++ helper for the CSE498 model API.

Mirrors the logic used in the Streamlit tester (`app.py`) so the web UI and
the Streamlit app produce comparable explanations.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _denorm_image(x_3chw: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    arr = x_3chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(std).reshape(1, 1, 3) + np.array(mean).reshape(1, 1, 3)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _get_cam_target_layer(model: nn.Module):
    """Pick a sensible last spatial layer for each supported architecture.

    Returns (layer, reshape_transform). ``reshape_transform`` is None for CNNs
    and a function for ViT (so token features become a [B,C,H,W] map).
    """
    if hasattr(model, "layer4"):  # torchvision ResNet
        return model.layer4[-1], None
    if hasattr(model, "features"):  # EfficientNet, DenseNet, MobileNet
        try:
            return model.features[-1], None
        except Exception:
            pass
    if hasattr(model, "blocks"):  # timm ViT family
        layer = model.blocks[-1].norm1

        def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor[:, 1:, :]
            h = int(tensor.shape[1] ** 0.5)
            return (
                tensor.reshape(tensor.size(0), h, h, tensor.size(2)).permute(0, 3, 1, 2)
            )

        return layer, reshape_transform
    raise ValueError("Could not infer Grad-CAM target layer for this architecture.")


def compute_gradcam_overlay_png(
    model: nn.Module,
    x_1chw: torch.Tensor,
    pred_idx: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> bytes:
    """Run Grad-CAM++ for the chosen class and return a PNG-encoded overlay."""
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer, reshape_transform = _get_cam_target_layer(model)
    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[layer],
        reshape_transform=reshape_transform,
    )
    gray = cam(
        input_tensor=x_1chw,
        targets=[ClassifierOutputTarget(int(pred_idx))],
    )[0]

    original = _denorm_image(x_1chw[0], mean, std)
    overlay = show_cam_on_image(original, gray, use_rgb=True)

    img = Image.fromarray(overlay)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
