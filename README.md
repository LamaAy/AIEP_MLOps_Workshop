<div align="center">
  <img src="assets/academy_banner.png" alt="Olo Academy — MLOps Workshop" width="100%" />

# MLOps Workshop
### Lab 01 — Train, Evaluate, and Save a Deep Learning Model

**PyTorch · Fashion-MNIST · CNN · Git & GitHub**
</div>

## Learning objectives
By the end of this exercise, you will be able to:
1. Clone a GitHub repository and set up a Python environment.
2. Train a convolutional neural network (CNN) using PyTorch.
3. Evaluate the model on a separate test split.
4. Save a **state dictionary** checkpoint (`.pth`) and reload it for inference.
5. Commit and push source-code changes, without committing model weights or downloaded datasets.

> **Scope:** This repository contains **Lab 01 only**. MLflow, APIs, Docker, CI/CD, and monitoring will come in later labs.

## Project layout
```text
AIEP_MLOps_Workshop/
├── assets/academy_banner.png
├── checkpoints/          # generated checkpoints (ignored by Git)
├── model.py              # CNN architecture
├── train.py              # train + evaluate + save checkpoint
├── evaluate.py           # load checkpoint + test accuracy
├── predict.py            # predict on one Fashion-MNIST sample
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Fork and clone
Fork the instructor repository on GitHub, then clone **your fork**:
```bash
git clone https://github.com/LamaAy/AIEP_MLOps_Workshop.git
cd AIEP_MLOps_Workshop
```
If working from the downloaded ZIP instead, extract it and enter the folder.

## 2. Install dependencies
Python **3.10+** is recommended.
```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If your machine needs a specific PyTorch CUDA build, follow the [official PyTorch install selector](https://pytorch.org/get-started/locally/). The default configuration also works on CPU.

## 3. Train the CNN
```bash
python train.py --epochs 2 --batch-size 128
```
The first run automatically downloads Fashion-MNIST into `data/`. The training script prints test accuracy and saves `checkpoints/model.pth`.

For a quick smoke test:
```bash
python train.py --epochs 1 --max-train-batches 3 --max-test-batches 2
```
This quick run **does not** produce meaningful accuracy; it only checks the pipeline.

## 4. Reload and evaluate
```bash
python evaluate.py --checkpoint checkpoints/model.pth
```

## 5. Run one prediction
```bash
python predict.py --checkpoint checkpoints/model.pth --index 0
```

## 6. Inspect your checkpoint
The checkpoint contains `model_state_dict`, `epoch`, `test_accuracy`, and `class_names`. We save weights rather than serializing the entire Python model, so the architecture is defined explicitly in `model.py`. Only load checkpoints from sources you trust.

## 7. Git & GitHub exercise
```bash
git status
git add README.md model.py train.py evaluate.py predict.py requirements.txt .gitignore assets/academy_banner.png
git commit -m "Complete Lab 01: train and checkpoint a CNN"
git push origin main
```
If the default branch on your fork is not `main`, use its actual branch name instead. **Do not add** `data/`, `.venv/`, or `checkpoints/*.pth`.

### Student tasks
- [ ] Run training and record your test accuracy.
- [ ] Change `--epochs` or `--batch-size` and observe the result.
- [ ] Explain why evaluation must use held-out data.
- [ ] Reload `model.pth` and make one prediction.
- [ ] Verify that `git status` does **not** show the dataset or checkpoint.
- [ ] Push a commit to your fork.

### Submission
Submit your **GitHub fork URL**, your **test accuracy**, and a short answer to: *Why should model weights and large datasets usually be managed separately from Git source history?*

## Troubleshooting
- **Slow training?** Reduce epochs or use the smoke-test flags; CPU is supported.
- **Download fails?** Verify network access to the Fashion-MNIST dataset source.
- **`ModuleNotFoundError`?** Activate `.venv` and run `pip install -r requirements.txt`.
- **GPU unavailable?** The scripts automatically use CPU when CUDA/MPS are unavailable.

## Next lab
Experiment tracking and model versioning with MLflow (separate exercise).
