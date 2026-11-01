<div align="center">
  <img src="assets/academy_banner.png" alt="Olo Academy — MLOps Workshop" width="100%" />

# MLOps Workshop
**PyTorch · MLflow · FastAPI · Docker · GitHub Actions**
</div>

A hands-on journey from **training** to **experiment tracking**, **model serving**, **containerization**, and **continuous integration** using Fashion-MNIST.

## Activities

| Lab | Topic | Outcome |
|---|---|---|
| [01 — Model Training](activity_01_training/README.md) | PyTorch CNN | Train and save **five** experiments as `.pth` checkpoints |
| [02 — Experiment Tracking](activity_02_mlflow/README.md) | MLflow | Load five checkpoints, compute rich evaluation metrics, compare runs — **no training** |
| [03 — API Serving](activity_03_fastapi/README.md) | FastAPI | Serve the CNN through HTTP endpoints |
| [04 — Containerization](activity_04_docker/README.md) | Docker | Package and run the inference service |
| [05 — Continuous Integration](activity_05_cicd/README.md) | GitHub Actions | Run automated API tests |

## Get started

```bash
git clone https://github.com/LamaAy/AIEP_MLOps_Workshop.git
cd AIEP_MLOps_Workshop
python -m venv .venv
source .venv/bin/activate
```

Follow the installation instructions in [Activity 01](activity_01_training/README.md). For Linux machines with limited space, consult the [PyTorch install selector](https://pytorch.org/get-started/locally/) for CPU-only packages before installing large CUDA dependencies.

## Workflow

**Activity 01:** Five training configurations → five saved checkpoints in `checkpoints/activity_02/`.

**Activity 02:** Read the same five checkpoints → evaluate them → log metrics and artifacts in MLflow.

**Activities 03–04:** Use `checkpoints/model.pth` from Activity 01's optional single-model command; see the relevant lab instructions.

**Important:** Run all commands from the repository root. Downloaded datasets, local MLflow databases and trained checkpoints are not committed to GitHub.
