# Activity 02 — Experiment Tracking with MLflow

**Theory → Practice:** After learning about parameters, metrics, artifacts and runs, students track the *same FashionCNN* introduced in Activity 01.

## Learning objectives
- Create an MLflow experiment and multiple runs.
- Log learning rate, batch size, train loss, test loss and test accuracy.
- Inspect the per-epoch metric curves and compare runs in the MLflow UI.
- Save a PyTorch checkpoint as an MLflow artifact (without committing large files to Git).

## Setup
From the **repository root**, install the additional package:

```bash
python -m pip install -r activity_02_mlflow/requirements.txt
```

## Lab steps

**1. Train and track three configurations.** All runs use the same seed, epoch count and data to make the comparison useful.

```bash
python activity_02_mlflow/train_mlflow.py --epochs 2 --batch-size 128 --lr 0.01 --run-name run_01
python activity_02_mlflow/train_mlflow.py --epochs 2 --batch-size 128 --lr 0.001 --run-name run_02
python activity_02_mlflow/train_mlflow.py --epochs 2 --batch-size 128 --lr 0.0001 --run-name run_03
```

**2. Launch the MLflow interface** from the repository root:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open **http://127.0.0.1:5000** in your browser. Find `FashionMNIST_CNN`, select the three runs, compare `test_accuracy`, and inspect `train_loss` across epochs.

**3. Answer:** Which run obtained the highest accuracy? Was it also the lowest loss? Which hyperparameter was changed? Were these results generated on a held-out test set or a validation set? What would change in a more rigorous production experiment?

## Optional quick smoke test

```bash
python activity_02_mlflow/train_mlflow.py --epochs 1 --max-train-batches 2 --max-test-batches 2 --run-name smoke_test
```

A smoke test validates that the pipeline runs; its accuracy is **not meaningful** and should not be used to select a model.

## Expected outputs
- Three runs in the MLflow UI, with hyperparameters and metric histories.
- A saved checkpoint in `checkpoints/activity_02/<run_id>/model.pth`.
- An MLflow artifact copy of each checkpoint.

## Instructor notes
- This introductory lab evaluates on Fashion-MNIST's test split to remain compatible with Activity 01. **Repeatedly selecting the best hyperparameters on the test set leaks evaluation information.** For a realistic workflow, create a validation split for tuning and reserve the test set for one final assessment.
- Track the same architecture across experiments; the hyperparameter differences are intentional.
- Local MLflow run data (`mlruns/`) and checkpoints are ignored by Git.
- Do not expose the local MLflow UI to the public internet.
