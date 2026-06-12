from __future__ import annotations

import io
from typing import Sequence

from PIL import Image
import torch
from torchvision import transforms


def read_image_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def preprocess_image(
    pil_img: Image.Image,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return tfm(pil_img).unsqueeze(0)

