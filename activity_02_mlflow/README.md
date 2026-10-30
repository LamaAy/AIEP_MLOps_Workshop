# Activity 02 — Experiment Tracking with MLflow

Compare three pretrained **Fashion-MNIST CNN checkpoints** from Activity 01. Students do not retrain models.

## Learning goals
- Register existing checkpoints as separate MLflow runs.
- Compare accuracy, cross-entropy loss, precision, recall, F1-score, and inference latency.
- Explore per-class scores, confusion matrices, and checkpoint artifacts.

## 1. Setup
Run these commands from the **repository root**:

```bash
python -m pip install -r requirements.txt
python -m pip install -r activity_02_mlflow/requirements.txt
```

## 2. Instructor preparation (once, before class)
Activity 01's `checkpoints/model.pth` must already exist. Prepare two additional **real** trained models, while retaining the original checkpoint:

```bash
python activity_02_mlflow/prepare_models.py --epochs 2
```

This creates three files in `checkpoints/activity_02/`. Share these checkpoints with students separately; checkpoints are not committed to Git.

## 3. Student exercise — evaluate and log (no training)
Place the three provided `.pth` files in `checkpoints/activity_02/`, then run:

```bash
python activity_02_mlflow/compare_models.py
```

## 4. View and compare runs

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open **http://127.0.0.1:5001** → **Model training** → **FashionMNIST_CNN_Comparison**. Select the three runs and compare their metrics and artifacts.

## Student questions
1. Which checkpoint has the highest macro F1-score? Is it also the most accurate?
2. Which classes are most frequently confused?
3. How does test loss compare across the three runs?
4. What are the limitations of comparing inference latency across machines?

**Output:** Three MLflow runs, numeric metrics, per-class metrics, confusion matrix CSVs, classification report JSONs, and saved checkpoint artifacts.
