# Activity 02 — Experiment Tracking with MLflow

Compare five trained **Fashion-MNIST CNN models** using MLflow without retraining.

## Learning Objectives

By the end of this activity, you will be able to:

1. Track machine learning experiments using MLflow.
2. Load and evaluate saved model checkpoints.
3. Compare accuracy, loss, precision, recall, F1-score, and inference latency.
4. Analyze confusion matrices and per-class performance.
5. Identify the best-performing model using evaluation metrics.

## 1. Setup

Open a terminal (**Terminal 1**) and navigate to the repository root:

```bash
cd ~/AIEP_MLOps_Workshop
```

Activate your virtual environment:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r activity_02_mlflow/requirements.txt
```

## 2. Verify Saved Checkpoints

The five models should already have been trained and saved during Activity 01.

Check that the checkpoint directory exists:

```bash
ls -lh checkpoints/activity_02/
```

You should see five `.pth` files corresponding to the five training experiments.

If the files are missing, complete Activity 01 first:

```bash
python activity_01_training/train_five.py
```

**Note:** Activity 02 does not retrain models. It only loads and evaluates existing checkpoints.

## 3. Start the MLflow Server — Terminal 1

From the repository root, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Keep **Terminal 1 open** while using MLflow.

Open your browser and navigate to:

http://127.0.0.1:5001

Initially, you may see only the **Default** experiment. This is normal before logging any runs.

## 4. Open a New Terminal — Terminal 2

Do not close Terminal 1.

Open a **second terminal window or tab**.

Navigate to the repository:

```bash
cd ~/AIEP_MLOps_Workshop
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Verify the five checkpoints:

```bash
ls -lh checkpoints/activity_02/
```

Now run the comparison:

```bash
python activity_02_mlflow/compare_models.py
```

The script will:

1. Load the five trained CNN checkpoints.
2. Evaluate each model on Fashion-MNIST.
3. Calculate performance metrics.
4. Record the results as separate MLflow runs.
5. Save evaluation reports and artifacts.

No training takes place during this step.

## 5. View and Compare Experiments

Return to your browser and refresh:

http://127.0.0.1:5001

Open the experiment:

**FashionMNIST_CNN_Comparison**

You should now see five runs, one for each checkpoint.

Select the runs to compare their metrics.

### Evaluation Metrics

The comparison includes:

- Accuracy and balanced accuracy
- Cross-entropy loss
- Precision, recall, and F1-score (macro and weighted)
- Per-class performance
- ROC-AUC
- Cohen's kappa and Matthews correlation coefficient
- Confidence and calibration metrics
- Inference latency and throughput

Explore the saved artifacts, including confusion matrices and classification reports.

## 6. Student Questions

1. Which experiment achieves the highest accuracy?
2. Which model has the highest macro F1-score?
3. How does changing the learning rate affect model performance?
4. Does increasing training epochs improve accuracy?
5. Which classes are most frequently confused?
6. Does the most accurate model also have the lowest inference latency?
7. Why should we compare multiple metrics instead of relying on accuracy alone?

## Expected Output

After completing this activity, you should have:

- Five MLflow experiment runs.
- A comparison of model evaluation metrics.
- Per-class classification reports.
- Confusion matrices.
- Saved checkpoint artifacts.
- An understanding of how MLflow supports experiment tracking and model comparison.

## 7. Stop MLflow

When finished, return to **Terminal 1** and press:

```text
Ctrl + C
```

This stops the MLflow server.

Your experiment results remain saved in the local MLflow tracking database.
