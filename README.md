<div align="center">
  <img src="assets/academy_banner.png" alt="Olo Academy — MLOps Workshop" width="100%" />

# MLOps Workshop
### Hands-on Labs — Training, Tracking, Serving, Docker, and CI/CD

**PyTorch · Fashion-MNIST · CNN · Git & GitHub**
</div>

## Learning objectives
By the end of Lab 01, you will be able to:
1. Clone a GitHub repository and set up a Python environment.
2. Train a convolutional neural network (CNN) using PyTorch.
3. Evaluate the model on a separate test split.
4. Save a PyTorch checkpoint (`.pth`) containing the model state dictionary and reload it for inference.
5. Commit and push source-code changes without committing model weights or downloaded datasets.

> **Repository scope:** Five hands-on activities covering model training, experiment tracking, API serving, containerization, and continuous integration.

## Project layout
```text
AIEP_MLOps_Workshop/
├── assets/
│   └── academy_banner.png
├── activity_01_training/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── model.py
│   └── utils.py
├── activity_02_mlflow/
├── activity_03_fastapi/
├── activity_04_docker/
├── activity_05_cicd/
├── checkpoints/           # generated model checkpoints
├── requirements.txt
└── README.md
```

**Run all commands below from the repository root** (`AIEP_MLOps_Workshop/`).

## 1. Fork and clone
Fork the instructor repository on GitHub, then clone **your fork**:
```bash
git clone https://github.com/LamaAy/AIEP_MLOps_Workshop.git
cd AIEP_MLOps_Workshop
```
If working from a downloaded ZIP, extract it and enter the folder instead.

## 2. Install dependencies
Use a Python version supported by your selected PyTorch build.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Low-disk-space / CPU-only setup:** On Linux, the default PyTorch installation can download large NVIDIA CUDA libraries. If you do not need GPU training, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for a CPU-only build compatible with your Python version. Install matching `torch` and `torchvision` versions, then install the other requirements. Avoid reinstalling GPU builds unintentionally.

## 3. Train the CNN

```bash
python activity_01_training/train.py --epochs 2 --batch-size 128
```

The first run downloads Fashion-MNIST if needed. The training script reports test accuracy and saves `checkpoints/model.pth`, relative to the repository root.

For a quick smoke test:

```bash
python activity_01_training/train.py --epochs 1 --max-train-batches 3 --max-test-batches 2
```

This quick run **does not** produce meaningful accuracy; it only checks the pipeline.

## 4. Reload and evaluate

```bash
python activity_01_training/evaluate.py --checkpoint checkpoints/model.pth
```

## 5. Run one prediction

```bash
python activity_01_training/predict.py --checkpoint checkpoints/model.pth --index 0
```

## Activity 02 — Experiment Tracking with MLflow

[Open Activity 02](activity_02_mlflow/README.md) — Compare three pretrained CNN checkpoints without retraining.

## Activity 03 — Model Serving with FastAPI

[Open Activity 03](activity_03_fastapi/README.md) — Serve the FashionCNN as a prediction API, test HTTP endpoints, and run inference without retraining.

## Activity 04 — Docker

[Start Activity 04](activity_04_docker/README.md) — Package the prediction service into a container.

## Activity 05 — CI with GitHub Actions

[Start Activity 05](activity_05_cicd/README.md) — Automatically test the FastAPI inference service on GitHub.
