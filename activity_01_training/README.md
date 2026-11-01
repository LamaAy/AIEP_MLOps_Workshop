# Activity 01 — Train and Save Five CNN Experiments

Train five **real, distinct Fashion-MNIST CNN models** with PyTorch. The models share the same architecture and dataset, but use different learning rates, batch sizes or training epochs. Each trained checkpoint will be reused by Activity 02 — **no retraining in Activity 02**.

## Learning goals
- Train, evaluate, save and reload a model.
- Observe the effects of learning rate, batch size and epoch count.
- Save model weights and experiment metadata to `.pth` checkpoints.
- Create five model checkpoints for MLflow comparison.

## 1. Setup
Run from the repository root (`AIEP_MLOps_Workshop/`):

```bash
python -m pip install -r requirements.txt
```

On machines with limited disk space, use an appropriate **CPU-only PyTorch build** following the [official installer](https://pytorch.org/get-started/locally/) instead of inadvertently downloading multiple large CUDA packages.

## 2. Train the five experiments

```bash
python activity_01_training/train_five.py
```

| Experiment | Epochs | Batch size | Learning rate | What changes? |
|---|---:|---:|---:|---|
| `cnn_01_baseline` | 2 | 128 | 0.001 | Baseline |
| `cnn_02_low_lr` | 2 | 128 | 0.0003 | Lower learning rate |
| `cnn_03_high_lr` | 2 | 128 | 0.003 | Higher learning rate |
| `cnn_04_small_batch` | 2 | 64 | 0.001 | Smaller batch |
| `cnn_05_longer_train` | 4 | 128 | 0.001 | More epochs |

All experiments use seed `42`, Adam and the same FashionCNN architecture. Runs train sequentially; the first run downloads the dataset if needed.

**Saved outputs:**

```text
checkpoints/
├── model.pth                       # optional single-model checkpoint for Activities 03/04
└── activity_02/
    ├── cnn_01_baseline.pth
    ├── cnn_02_low_lr.pth
    ├── cnn_03_high_lr.pth
    ├── cnn_04_small_batch.pth
    └── cnn_05_longer_train.pth
```

Each `.pth` file includes trained weights, epoch count, hyperparameters, class names and test accuracy. Do not commit checkpoints to GitHub.

## 3. Check the five checkpoints

```bash
ls -lh checkpoints/activity_02/*.pth
```

## 4. Evaluate and predict with one checkpoint

```bash
python activity_01_training/evaluate.py --checkpoint checkpoints/activity_02/cnn_01_baseline.pth
python activity_01_training/predict.py --checkpoint checkpoints/activity_02/cnn_01_baseline.pth --index 0
```

## 5. Optional: smoke test (not for comparing models)

```bash
python activity_01_training/train_five.py --max-train-batches 3 --max-test-batches 2
```

**Warning:** This writes five *incompletely trained* checkpoints to the same paths. After a smoke test, rerun the normal `train_five.py` command before Activity 02.

## 6. Optional: single-model training for API and Docker

```bash
python activity_01_training/train.py --epochs 2 --batch-size 128
```

This produces `checkpoints/model.pth`, which Activities 03 and 04 expect. The five comparison checkpoints do not replace this path automatically.

## Student questions
1. Why is it important to change one hyperparameter at a time?
2. Which checkpoint has the lowest test loss and the highest accuracy?
3. Why can a higher learning rate hurt or help training?
4. What is saved in `model_state_dict`? What is not saved?
5. Why should these checkpoints be kept outside Git?

**Next:** [Activity 02 — Experiment Tracking with MLflow](../activity_02_mlflow/README.md).
