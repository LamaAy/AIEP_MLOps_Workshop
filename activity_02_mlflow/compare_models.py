"""Register and compare existing FashionCNN checkpoints without any training."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'activity_01_training'))
from model import CLASS_NAMES, FashionCNN
from utils import data_loader, get_device


def evaluate(path, batch_size, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError(f'Checkpoint missing model_state_dict: {path}')
    model = FashionCNN().to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    loader = data_loader(train=False, batch_size=batch_size)
    labels, predictions, all_probs = [], [], []
    total_loss, n = 0.0, 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_loss += nn.functional.cross_entropy(logits, targets.to(device), reduction='sum').item()
            n += len(targets)
            labels.extend(targets.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            all_probs.append((elapsed, len(targets)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(10)), zero_division=0)
    metrics = {'test_accuracy': float(accuracy_score(labels, predictions)),
               'test_loss': float(total_loss / n),
               'precision_macro': float(precision.mean()),
               'recall_macro': float(recall.mean()),
               'f1_macro': float(f1.mean()),
               'inference_ms_per_image': float(1000 * sum(t for t, count in all_probs) / n)}
    for avg in ('weighted',):
        p, r, f, _ = precision_recall_fscore_support(labels, predictions, average=avg, zero_division=0)
        metrics.update({f'precision_{avg}': float(p), f'recall_{avg}': float(r), f'f1_{avg}': float(f)})
    cm = confusion_matrix(labels, predictions, labels=list(range(10)))
    report = classification_report(labels, predictions, labels=list(range(10)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    return ckpt, metrics, cm, report


def main():
    parser = argparse.ArgumentParser(description='Compare existing checkpoints in MLflow; NO TRAINING')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'checkpoints' / 'activity_02')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--experiment', default='FashionMNIST_CNN_Comparison')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('batch-size must be positive')
    paths = sorted(args.models_dir.glob('*.pth'))
    if len(paths) != 3:
        parser.error(f'Expected exactly 3 genuine .pth checkpoints in {args.models_dir}; found {len(paths)}. See README.')
    # SQLite, compatible with modern MLflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)
    device = get_device()
    print(f'Device: {device}; checkpoints: {[p.name for p in paths]}')
    for path in paths:
        ckpt, metrics, cm, report = evaluate(path, args.batch_size, device)
        with mlflow.start_run(run_name=path.stem) as run:
            mlflow.log_params({'architecture': 'FashionCNN', 'dataset': 'Fashion-MNIST',
                               'checkpoint_name': path.name, 'evaluation_batch_size': args.batch_size,
                               'retrained': False, 'split': 'test'})
            for key in ('epoch', 'learning_rate', 'batch_size', 'seed'):
                if key in ckpt and isinstance(ckpt[key], (float, int, str, bool)):
                    mlflow.log_param(f'training_{key}', ckpt[key])
            mlflow.log_metrics(metrics)
            for class_name, scores in report.items():
                if class_name in CLASS_NAMES:
                    class_id = CLASS_NAMES.index(class_name)
                    for metric_name in ('precision', 'recall', 'f1-score'):
                        mlflow.log_metric(f'class_{class_id}_{metric_name.replace("-", "_")}', float(scores[metric_name]))
            mlflow.log_artifact(str(path), artifact_path='checkpoints')
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                td = Path(td)
                (td / 'classification_report.json').write_text(json.dumps(report, indent=2))
                with (td / 'confusion_matrix.csv').open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['actual/predicted', *CLASS_NAMES])
                    for name, row in zip(CLASS_NAMES, cm):
                        writer.writerow([name, *row.tolist()])
                mlflow.log_artifacts(str(td), artifact_path='evaluation')
            print(f'{path.name}: accuracy={metrics["test_accuracy"]:.2%}, '
                  f'F1-macro={metrics["f1_macro"]:.4f}, loss={metrics["test_loss"]:.4f}; run={run.info.run_id}')


if __name__ == '__main__':
    main()
