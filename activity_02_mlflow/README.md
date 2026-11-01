# Activity 02 — Compare Five Trained Models with MLflow

Load the **five checkpoints produced in Activity 01**, evaluate them on the Fashion-MNIST test split, and log every run to MLflow. **No training happens in this activity.**

## Learning goals
- Track and compare five experiments using the same test data.
- Interpret class-level and aggregate classification metrics.
- Inspect confusion matrices, probability calibration, and latency.
- Compare hyperparameters against test results in the MLflow UI.

## 1. Install dependencies
Run from the **repository root**:

```bash
python -m pip install -r activity_02_mlflow/requirements.txt
```

Make sure `torch`, `torchvision`, and `numpy` from Activity 01 are already installed. If not, complete the Activity 01 setup first.

## 2. Check your five saved models

```bash
ls -lh checkpoints/activity_02/*.pth
```

The directory must contain **exactly five `.pth` files** from Activity 01. If they are missing:

```bash
python activity_01_training/train_five.py
```

This is an Activity 01 training command; Activity 02 itself **does not train**.

## 3. Evaluate and log all five runs

```bash
python activity_02_mlflow/compare_models.py
```

The script reloads each checkpoint and records metrics and artifacts in the local SQLite tracking database `mlflow.db`.

### Metrics

| Category | Logged metrics |
|---|---|
| Overall | Accuracy, error rate, balanced accuracy, cross-entropy loss |
| Precision/recall | Macro and weighted precision/recall/F1; micro F1 |
| Agreement | Cohen's kappa, Matthews correlation coefficient |
| Ranking | Top-2 and top-3 accuracy, one-vs-rest ROC-AUC (macro/weighted) |
| Probabilities | Log loss, multiclass Brier score, 15-bin Expected Calibration Error (ECE), confidence for correct/wrong predictions |
| Performance | Inference milliseconds/image, throughput (images/second), parameter count |
| Per class | Precision, recall, F1, specificity and one-vs-rest ROC-AUC for each of 10 classes |

The script also saves **confusion matrix CSV, per-class CSV, calibration CSV, per-example predictions with class probabilities, and classification report JSON** for each run. Inference timing covers **forward-pass execution only**, excludes dataset loading/preprocessing, and varies with device/batch size; it is not a standardized end-to-end benchmark.

## 4. Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open **http://127.0.0.1:5001** → **Model training** → **FashionMNIST_CNN_Comparison**. Select five runs and compare metrics and artifacts.

## 5. Review the exported summary

```bash
ls -lh checkpoints/activity_02_summary.csv
```

## Student questions
1. Which run achieves the highest accuracy and which achieves the highest macro F1?
2. Do precision, recall and F1 tell the same story for `Shirt` and `T-shirt/top`?
3. Does longer training guarantee the best loss, accuracy or calibration?
4. Why does test accuracy not tell us whether a model's probabilities are well calibrated?
5. Why might inference latency differ between CPU and GPU?
6. Are these five models architecturally different, or do their training configurations differ?

**Expected result:** Five MLflow runs, dozens of scalar metrics per run, ten-class metrics and structured evaluation artifacts. No retraining.
