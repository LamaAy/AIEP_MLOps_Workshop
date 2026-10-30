# Activity 01 — Deep Learning Model Training

## Objective
Train, evaluate, and save a CNN using PyTorch.

## Dataset
Fashion-MNIST

## Tasks
1. Load the dataset.
2. Train the CNN.
3. Evaluate its accuracy.
4. Save the model checkpoint.
5. Load the model and run inference.

## Commands

Run from the repository root:

    python activity_01_training/train.py --epochs 2 --batch-size 128
    python activity_01_training/evaluate.py --checkpoint checkpoints/model.pth
    python activity_01_training/predict.py --checkpoint checkpoints/model.pth --index 0

## Expected Results
- Trained CNN
- Evaluation accuracy
- Saved checkpoint
- Sample prediction
