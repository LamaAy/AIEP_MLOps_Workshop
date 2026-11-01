# Activity 02 — Experiment Tracking with MLflow

Compare three trained **Fashion-MNIST CNN models** using MLflow without retraining.

## Learning Objectives
By the end of this activity, you will be able to:

1. Track machine learning experiments using MLflow.
2. Load and evaluate pretrained model checkpoints.
3. Compare accuracy, loss, precision, recall, F1-score, and inference latency.
4. Analyze confusion matrices and per-class performance.
5. Identify the best-performing model based on evaluation metrics.

## 1. Setup

Open a terminal and navigate to the repository root:

```bash
cd AIEP_MLOps_Workshop
```

Activate your virtual environment and install the dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r activity_02_mlflow/requirements.txt
```

## 2. Prepare the Models

This activity uses three pretrained CNN checkpoints.

Make sure the following directory contains the three provided `.pth` files:

```text
checkpoints/
└── activity_02/
    ├── model_1.pth
    ├── model_2.pth
    └── model_3.pth
```

The filenames above are illustrative; use the actual checkpoint filenames provided.

**Note:** No model training is required in this activity.

## 3. Evaluate and Track Experiments

Run the comparison script from the repository root:

```bash
python activity_02_mlflow/compare_models.py
```

The script evaluates the three models and records their performance in MLflow.

Tracked metrics include:

- Accuracy and cross-entropy loss
- Precision, recall, and F1-score
- Inference latency
- Per-class performance

## 4. Open MLflow Dashboard

Start the MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open your browser:

http://127.0.0.1:5001

Navigate to **FashionMNIST_CNN_Comparison**, select the three runs, and compare their metrics and artifacts.

Explore the confusion matrices, classification reports, and saved checkpoints.

## 5. Student Questions

1. Which model achieves the highest macro F1-score? Is it also the most accurate?
2. Which Fashion-MNIST classes are most frequently confused?
3. How does cross-entropy loss differ across the three models?
4. Does the model with the highest accuracy also have the lowest inference latency?
5. Why is experiment tracking important in machine learning?

## Expected Output

After completing the activity, you should have:

- Three MLflow experiment runs.
- A comparison of model evaluation metrics.
- Confusion matrices and per-class performance reports.
- Saved model checkpoint artifacts.
- An understanding of how MLflow supports model comparison and experiment tracking.
