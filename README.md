# Explainable AI-Based Eye Disease Classification Using Retinal Images

**CSE498 university research · North South University, Dhaka**

A research implementation for multi-class eye-disease image classification using pretrained CNNs, a vision transformer, custom neural networks, **Grad-CAM++**, and **federated-learning simulations**. The project connects notebook experiments and saved model bundles to a **Streamlit interface** and a **FastAPI inference service** that can be used by web and mobile clients.

[Explore the project and image gallery](https://shahriar811.github.io/mdshahriarhossain.github.io/projects/retinal-eye-disease.html) · [API documentation](API/README.md) · [Public HTTPS setup](API%20Global/README.md)

## At a glance

| Baseline study | Recorded value |
| --- | ---: |
| Indexed augmented images | **16,242** |
| Image classes | **10** |
| Train / validation / test | **11,369 / 2,436 / 2,437** |
| Best backbone in the saved comparison | **DenseNet121** |
| DenseNet121 test accuracy | **89.82%** |
| DenseNet121 weighted F1 | **89.74%** |
| Lightweight OurCNN parameters | **122,972** |
| Federated strategies | **FedAvg and FedProx** |

These values describe saved experiments in `CSE498V2.ipynb` and the separate lightweight-CNN notebook. The MSCA-Net notebook uses a different data preparation and training configuration; its setup is described separately below.

## Research workflow

1. Index images and labels, inspect class distributions, and create stratified splits.
2. Resize and normalize images, apply training augmentation, and handle class imbalance.
3. Train and compare pretrained backbones and custom networks.
4. Evaluate accuracy, precision, recall, weighted F1, and confusion matrices.
5. Use Grad-CAM++ to visualize image regions influencing model predictions.
6. Run local federated simulations with FedAvg and FedProx.
7. Serve compatible model bundles through Streamlit or FastAPI.

![Project architecture diagram](Project%20Diagram.png)

The diagram illustrates the overall research design. The endpoint table below documents the routes implemented in the current API, including `/predict_xai` for explanations.

## Models and experiments

| Model / experiment | Role | Notebook |
| --- | --- | --- |
| ResNet50 | Pretrained CNN baseline | [CSE498V2.ipynb](CSE498V2.ipynb) |
| EfficientNet-B0 | Efficient pretrained CNN baseline | [CSE498V2.ipynb](CSE498V2.ipynb) |
| DenseNet121 | Best backbone in the saved baseline comparison | [CSE498V2.ipynb](CSE498V2.ipynb) |
| MobileNetV3-Small | Lightweight backbone; used in baseline federated experiments | [CSE498V2.ipynb](CSE498V2.ipynb) |
| ViT Small | Vision-transformer baseline | [CSE498V2.ipynb](CSE498V2.ipynb) |
| OurCNN | Custom lightweight CNN with 122,972 parameters | [Custom CNN(Light weight).ipynb](Custom%20CNN%28Light%20weight%29.ipynb) |
| MSCA-Net | Custom Multi-Scale Cross-Attention Network | [Custom CNN(Main).ipynb](Custom%20CNN%28Main%29.ipynb) |

**MSCA-Net** combines an EfficientNet-B3 backbone, channel/spatial attention, cross-scale projections, and multi-scale feature fusion. Its saved architecture output reports 11,959,512 total parameters. The notebook also includes a separate balancing step using `WeightedRandomSampler`, mixed-precision training, and federated experiment configuration.

Additional custom-model development is retained in [Custom CNN.ipynb](Custom%20CNN.ipynb).

## Dataset

The baseline notebook indexes the **Augmented Dataset** directory of the Eye Disease Image Dataset.

| Image class | Indexed images |
| --- | ---: |
| Central Serous Chorioretinopathy [Color Fundus] | 606 |
| Diabetic Retinopathy | 3,444 |
| Disc Edema | 762 |
| Glaucoma | 2,880 |
| Healthy | 2,676 |
| Macular Scar | 1,937 |
| Myopia | 2,251 |
| Pterygium | 102 |
| Retinal Detachment | 750 |
| Retinitis Pigmentosa | 834 |
| **Total** | **16,242** |

The baseline uses a stratified **70 / 15 / 15** train/validation/test split. The MSCA-Net notebook combines augmented and original data using its own setup; saved outputs record **20,510 training images** and **1,067 validation images**. These counts are separate experiment records and should not be merged with the baseline totals.

Counts describe indexed images, including augmented images. They do not establish the number of unique patients or a patient-level split. Download the dataset separately and update the notebook's dataset paths before training.

![Baseline image-class distribution](https://raw.githubusercontent.com/Shahriar811/mdshahriarhossain.github.io/main/images/projects/research/retinal-dataset-distribution.png)

## Recorded test results

| Model | Accuracy | Weighted precision | Weighted F1 |
| --- | ---: | ---: | ---: |
| DenseNet121 | **89.82%** | 89.87% | **89.74%** |
| ResNet50 | 89.66% | 89.74% | 89.57% |
| EfficientNet-B0 | 89.50% | 89.59% | 89.45% |
| MobileNetV3-Small | 86.79% | 86.98% | 86.61% |
| ViT Small | 82.36% | 82.92% | 82.15% |
| OurCNN | 76.24% | 77.35% | 75.48% |

The five backbone scores come from the saved `CSE498V2.ipynb` comparison on a **2,437-image test set**. OurCNN values come from the separate `Custom CNN(Light weight).ipynb` experiment, which also records a 2,437-image test split. The MSCA-Net experiment is not included in this baseline comparison.

OurCNN uses approximately **64.9× fewer parameters** than the DenseNet121 reference (122,972 versus 7,978,856), with the accuracy trade-off shown above.

![Recorded model comparison](https://raw.githubusercontent.com/Shahriar811/mdshahriarhossain.github.io/main/images/projects/research/retinal-model-performance.png)

[View source-labelled result data](https://shahriar811.github.io/mdshahriarhossain.github.io/projects/data/retinal-results.json)

## Explainable AI

**Grad-CAM++** produces attention heatmaps alongside model predictions. The notebook, Streamlit interface, and API contain explanation workflows. The heatmap visualizes model attention rather than providing a lesion segmentation or a clinical diagnosis.

![Original saved Grad-CAM++ example](https://raw.githubusercontent.com/Shahriar811/mdshahriarhossain.github.io/main/images/projects/research/retinal-gradcam-example.png)

## Federated learning

`CSE498V2.ipynb` simulates **three clients**, **three rounds**, and **one local epoch per round** with a MobileNetV3-Small backbone. It compares FedAvg against FedProx (`mu = 0.01`) without combining client training data into a central training pool.

| Strategy | Round 1 validation accuracy | Round 2 | Round 3 | Round 3 weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 66.34% | 75.45% | **79.15%** | 78.71% |
| FedProx | 66.42% | 75.33% | **78.98%** | 78.55% |

These are **validation scores**, not the held-out test scores in the model comparison. The clients are simulated locally in the notebook; this is not a deployed multi-institution federation.

## Repository structure

```text
CSE498V2.ipynb                 Baseline training, evaluation, XAI, and FL
Custom CNN(Light weight).ipynb Lightweight OurCNN experiment
Custom CNN(Main).ipynb         MSCA-Net custom-model experiment
Custom CNN.ipynb               Additional custom-model development
CSE498R models/                Saved checkpoints and inference bundles
app.py                        Streamlit model-testing interface
API/                          FastAPI server, requirements, and client example
API Global/                   Scripts and instructions for an HTTPS tunnel
Project Diagram.png           Research architecture diagram
```

The API supports mobile clients through HTTP endpoints. Android client source in the local development workspace is not currently included in this repository.

## Run the project

### 1. Clone and prepare an environment

```powershell
git clone https://github.com/Shahriar811/Explainable-AI-Based-Eye-Disease-Classification-Using-Retinal-Image.git
cd Explainable-AI-Based-Eye-Disease-Classification-Using-Retinal-Image
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r API/requirements.txt
```

The code uses PyTorch, torchvision, timm, Pillow, NumPy, FastAPI, and Grad-CAM++. Inference selects CUDA when available and otherwise runs on CPU.

### 2. Launch the FastAPI interface

```powershell
cd API
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) for the image-upload UI or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API reference. The default model directory is the repository's `CSE498R models/`; set the `MODEL_DIR` environment variable to use a different bundle directory.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/` | Browser image-upload interface |
| GET | `/models` | Discover available model bundles |
| POST | `/predict` | Predict a class from an uploaded image |
| POST | `/predict_xai` | Predict a class and return a Grad-CAM++ overlay |
| GET | `/docs` | Interactive API reference |

See [API/README.md](API/README.md) and [API/client_example.py](API/client_example.py) for serving and client details.

### 3. Launch the Streamlit tester

From the repository root:

```powershell
python -m pip install streamlit
python -m streamlit run app.py
```

Before launching, update the hardcoded `MODEL_DIR` in `app.py` to your local model directory. A portable replacement is:

```python
MODEL_DIR = Path(__file__).resolve().parent / "CSE498R models"
```

Open [http://localhost:8501](http://localhost:8501) to choose a compatible model bundle, upload an image, and inspect its prediction and optional heatmap.

**Bundle compatibility:** the current API loader supports the four standard CNN backbones plus ViT Base and ViT Small. The current Streamlit loader supports those CNNs and ViT Base. Custom OurCNN/MSCA-Net models require their notebook definitions or an added architecture loader; not every training checkpoint is directly supported by the serving interfaces.

### 4. Reproduce notebook experiments

```powershell
python -m pip install jupyter pandas scikit-learn matplotlib seaborn tqdm
jupyter notebook
```

Start with `CSE498V2.ipynb`. Update its `DATASET_DIR` and `MODEL_DIR`, then run setup, indexing, splits, training, evaluation, XAI, and federation in order. Use the custom notebooks for their respective model experiments; MSCA-Net detects dataset directories for supported environments and has its own configuration block.

For temporary public HTTPS access, follow [API Global/README.md](API%20Global/README.md). The tunnel points to a running local API; it is not permanent hosted deployment.

## Research context

Developed as part of **CSE498 research at North South University**. The repository preserves recorded experiments and deployment prototypes; reported metrics should be read with their dataset, split, and experiment source.

**Md. Shahriar Hossain** · [GitHub](https://github.com/Shahriar811) · [Portfolio](https://shahriar811.github.io/mdshahriarhossain.github.io/)
