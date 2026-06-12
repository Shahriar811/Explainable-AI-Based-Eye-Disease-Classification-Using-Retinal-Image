import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
import timm


st.set_page_config(page_title="CSE498R Model Tester", layout="wide")
st.title("CSE498R Eye Disease Model Tester")

MODEL_DIR = Path(r"D:\CSE Code\CSE498R\CSE498R models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model_by_arch(name: str, num_classes: int):
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
    raise ValueError(f"Unsupported architecture in bundle: {name}")


def unwrap_logits(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


@st.cache_resource(show_spinner=False)
def load_bundle(bundle_path: str):
    payload = torch.load(bundle_path, map_location="cpu")
    arch = payload["architecture"]
    class_names = payload["class_names"]
    image_size = int(payload.get("image_size", 224))
    mean = payload.get("mean", [0.485, 0.456, 0.406])
    std = payload.get("std", [0.229, 0.224, 0.225])
    model = create_model_by_arch(arch, len(class_names))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(DEVICE).eval()
    return model, arch, class_names, image_size, mean, std


def preprocess_image(pil_img: Image.Image, image_size: int, mean, std):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tfm(pil_img.convert("RGB")).unsqueeze(0)


def predict(model, pil_img, class_names, image_size, mean, std):
    x = preprocess_image(pil_img, image_size, mean, std).to(DEVICE)
    with torch.no_grad():
        out = unwrap_logits(model(x))
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
    return pred_idx, probs.cpu().numpy()


def get_cam_target_layer(model):
    if hasattr(model, "layer4"):
        return model.layer4[-1], None
    if hasattr(model, "features"):
        try:
            return model.features[-1], None
        except Exception:
            pass
    if hasattr(model, "blocks"):
        layer = model.blocks[-1].norm1

        def reshape_transform(tensor):
            tensor = tensor[:, 1:, :]
            h = int((tensor.shape[1]) ** 0.5)
            return tensor.reshape(tensor.size(0), h, h, tensor.size(2)).permute(0, 3, 1, 2)

        return layer, reshape_transform
    raise ValueError("Could not infer Grad-CAM target layer for this architecture.")


def denorm_image(x_3chw, mean, std):
    arr = x_3chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = arr * np.array(std).reshape(1, 1, 3) + np.array(mean).reshape(1, 1, 3)
    return np.clip(arr, 0, 1)


def run_gradcam(model, pil_img, pred_idx, image_size, mean, std):
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    x = preprocess_image(pil_img, image_size, mean, std).to(DEVICE)
    layer, reshape_transform = get_cam_target_layer(model)
    cam = GradCAMPlusPlus(model=model, target_layers=[layer], reshape_transform=reshape_transform)
    gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]
    original = denorm_image(x[0], mean, std)
    overlay = show_cam_on_image(original, gray, use_rgb=True)
    return original, overlay


if not MODEL_DIR.exists():
    st.error(f"Model directory does not exist: {MODEL_DIR}")
    st.stop()

bundle_paths = sorted(MODEL_DIR.glob("*_bundle.pth"))
if not bundle_paths:
    st.warning("No *_bundle.pth files found. Train models first using CSE498V2.ipynb.")
    st.stop()

selected_bundle = st.selectbox("Select model bundle", [str(p) for p in bundle_paths])
uploaded = st.file_uploader("Upload retinal image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
show_xai = st.checkbox("Show Grad-CAM++ XAI", value=True)

if uploaded is not None:
    pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    model, arch, class_names, image_size, mean, std = load_bundle(selected_bundle)
    pred_idx, probs = predict(model, pil_img, class_names, image_size, mean, std)

    st.subheader("Prediction")
    st.write(f"**Architecture:** {arch}")
    st.write(f"**Predicted class:** {class_names[pred_idx]}")
    st.write(f"**Confidence:** {float(probs[pred_idx]):.2%}")

    top_k = min(5, len(class_names))
    top_idx = np.argsort(-probs)[:top_k]
    top_data = [
        {"class": class_names[i], "probability": float(probs[i])}
        for i in top_idx
    ]
    st.dataframe(top_data, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded image", use_column_width=True)

    if show_xai:
        try:
            original, overlay = run_gradcam(model, pil_img, pred_idx, image_size, mean, std)
            with col2:
                st.image(overlay, caption="Grad-CAM++ overlay", use_column_width=True)
        except Exception as ex:
            st.warning(f"XAI failed for this model/image: {ex}")
    else:
        with col2:
            st.info("Enable 'Show Grad-CAM++ XAI' to view explanation.")

st.caption(f"Device: {DEVICE}")
