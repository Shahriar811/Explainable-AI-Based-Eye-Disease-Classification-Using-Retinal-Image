from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision
import timm


def create_model_by_arch(name: str, num_classes: int) -> torch.nn.Module:
    if name == "ResNet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "EfficientNet-B0":
        model = torchvision.models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "DenseNet121":
        model = torchvision.models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if name == "MobileNetV3-Small":
        model = torchvision.models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if name == "ViT Base":
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    if name == "ViT Small":
        model = timm.create_model("vit_small_patch16_224", pretrained=False)
        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture in bundle: {name}")


def unwrap_logits(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


@dataclass(frozen=True)
class BundleMeta:
    architecture: str
    class_names: List[str]
    image_size: int
    mean: Sequence[float]
    std: Sequence[float]


@dataclass
class LoadedBundle:
    model: torch.nn.Module
    meta: BundleMeta


def load_bundle(bundle_path: Path, device: torch.device) -> LoadedBundle:
    payload = torch.load(str(bundle_path), map_location="cpu")
    arch = payload["architecture"]
    class_names = list(payload["class_names"])
    image_size = int(payload.get("image_size", 224))
    mean = payload.get("mean", [0.485, 0.456, 0.406])
    std = payload.get("std", [0.229, 0.224, 0.225])

    model = create_model_by_arch(arch, len(class_names))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()

    return LoadedBundle(
        model=model,
        meta=BundleMeta(
            architecture=arch,
            class_names=class_names,
            image_size=image_size,
            mean=mean,
            std=std,
        ),
    )


def predict_proba(model: torch.nn.Module, x_1chw: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = unwrap_logits(model(x_1chw))
        probs = torch.softmax(out, dim=1)
    return probs

