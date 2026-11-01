# Activity 01 — Deep Learning Model Training

## Objective

Train a Convolutional Neural Network (CNN) using PyTorch, evaluate its performance, save the trained model, and reload it for inference.

## Dataset

**Fashion-MNIST** — A dataset of 70,000 grayscale images across 10 clothing categories.

## Tasks

1. Load and preprocess the Fashion-MNIST dataset.
2. Build and train a CNN using PyTorch.
3. Evaluate the model on the test dataset.
4. **Save the trained model as a `.pth` checkpoint.**
5. Reload the saved model and evaluate it.
6. Run inference on a sample image.

## 1. Train and Save the Model

Run from the repository root:

```bash
python activity_01_training/train.py --epochs 2 --batch-size 128
```

The script will automatically:

- Download Fashion-MNIST if necessary.
- Train the CNN for 2 epochs.
- Evaluate the trained model.
- Save the checkpoint to `checkpoints/model.pth`.

The saved checkpoint contains the model's learned weights, training epoch, test accuracy, and class names.

## 2. Verify the Saved Checkpoint

Check that the model file was created:

```bash
ls -lh checkpoints/model.pth
```

The checkpoint can now be reused without training the model again.

## 3. Reload and Evaluate

```bash
python activity_01_training/evaluate.py --checkpoint checkpoints/model.pth
```

This loads the saved weights and evaluates the model on the test dataset.

## 4. Run Inference

```bash
python activity_01_training/predict.py --checkpoint checkpoints/model.pth --index 0
```

The script predicts the clothing category of one Fashion-MNIST image.

## 5. Student Questions

1. What is the difference between model training and inference?
2. What information is stored inside a `.pth` checkpoint?
3. Why do we save a trained model instead of retraining it every time?
4. What is the purpose of `model.eval()`?
5. What is the difference between saving a model and exporting it to ONNX?

## Expected Results

By the end of this activity, you should have:

- A trained Fashion-MNIST CNN.
- Test accuracy results.
- A saved checkpoint at `checkpoints/model.pth`.
- A successful evaluation using the saved checkpoint.
- A prediction on a sample test image.

**Next: Activity 02 — Experiment Tracking with MLflow**

In Activity 02, you will compare multiple trained checkpoints, log evaluation metrics, and analyze model performance using MLflow.
