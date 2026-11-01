"""Evaluate five previously trained FashionCNN models; NO training here."""
import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_recall_fscore_support, roc_auc_score, top_k_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'activity_01_training'))
from model import CLASS_NAMES, FashionCNN
from utils import data_loader, get_device

N_CLASSES = len(CLASS_NAMES)


def expected_calibration_error(confidence, correct, bins=15):
    edges = np.linspace(0., 1., bins + 1)
    ece = 0.0
    table = []
    for i in range(bins):
        mask = (confidence >= edges[i]) & ((confidence <= edges[i+1]) if i == bins - 1 else (confidence < edges[i+1]))
        if not np.any(mask):
            table.append([i, float(edges[i]), float(edges[i+1]), 0, None, None])
            continue
        acc = float(np.mean(correct[mask]))
        conf = float(np.mean(confidence[mask]))
        ece += np.mean(mask) * abs(acc - conf)
        table.append([i, float(edges[i]), float(edges[i+1]), int(mask.sum()), acc, conf])
    return float(ece), table


def evaluate(path, batch_size, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(f'Invalid checkpoint: {path}')
    model = FashionCNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    loader = data_loader(train=False, batch_size=batch_size)
    y_true, probs, latency_s = [], [], []
    total_loss = 0.
    n = 0
    with torch.inference_mode():
        for images, labels in loader:
            inputs, targets = images.to(device), labels.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            batch = len(labels)
            latency_s.append((elapsed, batch))
            total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item()
            n += batch
            y_true.extend(labels.tolist())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not n:
        raise ValueError('Empty evaluation dataset')
    y = np.asarray(y_true, dtype=int)
    p = np.concatenate(probs)
    predicted = p.argmax(1)
    correct = (predicted == y)
    confidence = p.max(1)
    one_hot = np.eye(N_CLASSES)[y]
    cm = confusion_matrix(y, predicted, labels=list(range(N_CLASSES)))
    report = classification_report(y, predicted, labels=list(range(N_CLASSES)),
                                   target_names=CLASS_NAMES, zero_division=0, output_dict=True)
    ece, calibration = expected_calibration_error(confidence, correct)
    precisions, recalls, fscores, supports = precision_recall_fscore_support(
        y, predicted, labels=list(range(N_CLASSES)), zero_division=0)
    total_tp = np.diag(cm).sum()
    metrics = {
        'test_accuracy': float(accuracy_score(y, predicted)),
        'test_error_rate': float(1 - accuracy_score(y, predicted)),
        'test_loss': float(total_loss / n),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'precision_macro': float(np.mean(precisions)),
        'recall_macro': float(np.mean(recalls)),
        'f1_macro': float(np.mean(fscores)),
        'f1_micro': float(f1_score(y, predicted, average='micro')),
        'precision_weighted': float(np.average(precisions, weights=supports)),
        'recall_weighted': float(np.average(recalls, weights=supports)),
        'f1_weighted': float(np.average(fscores, weights=supports)),
        'cohen_kappa': float(cohen_kappa_score(y, predicted)),
        'matthews_corrcoef': float(matthews_corrcoef(y, predicted)),
        'top2_accuracy': float(top_k_accuracy_score(y, p, k=2, labels=np.arange(N_CLASSES))),
        'top3_accuracy': float(top_k_accuracy_score(y, p, k=3, labels=np.arange(N_CLASSES))),
        'log_loss': float(log_loss(y, p, labels=np.arange(N_CLASSES))),
        'brier_multiclass': float(np.mean(np.sum((p - one_hot)**2, axis=1))),
        'expected_calibration_error_15bins': ece,
        'mean_confidence': float(np.mean(confidence)),
        'mean_confidence_correct': float(np.mean(confidence[correct])) if correct.any() else 0.,
        'mean_confidence_wrong': float(np.mean(confidence[~correct])) if (~correct).any() else 0.,
        'inference_ms_per_image': float(1000 * sum(t for t, _ in latency_s) / n),
        'inference_images_per_second': float(n / sum(t for t, _ in latency_s)),
        'num_test_samples': float(n),
        'num_model_parameters': float(sum(z.numel() for z in model.parameters())),
    }
    # ROC-AUC needs both positive and negative cases for all classes.
    try:
        metrics['roc_auc_ovr_macro'] = float(roc_auc_score(y, p, multi_class='ovr', average='macro', labels=np.arange(N_CLASSES)))
        metrics['roc_auc_ovr_weighted'] = float(roc_auc_score(y, p, multi_class='ovr', average='weighted', labels=np.arange(N_CLASSES)))
    except ValueError:
        pass
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = n - tp - fn - fp
        specificity = float(tn / (tn + fp)) if tn + fp else 0.
        per_class.append({'class_id': i, 'class_name': name, 'support': int(supports[i]),
                          'precision': float(precisions[i]), 'recall': float(recalls[i]),
                          'f1': float(fscores[i]), 'specificity': specificity})
        metrics[f'class_{i}_precision'] = float(precisions[i])
        metrics[f'class_{i}_recall'] = float(recalls[i])
        metrics[f'class_{i}_f1'] = float(fscores[i])
        metrics[f'class_{i}_specificity'] = specificity
        try:
            metrics[f'class_{i}_roc_auc'] = float(roc_auc_score((y == i).astype(int), p[:, i]))
        except ValueError:
            pass
    return checkpoint, metrics, cm, report, per_class, calibration, y, predicted, p


def write_csv(path, headers, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Evaluate five saved models in MLflow, without retraining')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'checkpoints' / 'activity_02')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--experiment', default='FashionMNIST_CNN_Comparison')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be >= 1')
    paths = sorted(args.models_dir.glob('*.pth'))
    if len(paths) != 5:
        parser.error(f'Expected exactly 5 .pth checkpoints in {args.models_dir}; found {len(paths)}. Run Activity 01 train_five.py first.')
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)
    device = get_device()
    print(f'Device: {device} | Evaluating {len(paths)} checkpoints; training is disabled.')
    summary = []
    for path in paths:
        checkpoint, metrics, cm, report, per_class, calibration, y, predicted, p = evaluate(path, args.batch_size, device)
        with mlflow.start_run(run_name=path.stem) as run:
            params = {'architecture': 'FashionCNN', 'dataset': 'Fashion-MNIST',
                      'checkpoint_name': path.name, 'evaluation_batch_size': args.batch_size,
                      'split': 'test', 'retrained': False, 'device': str(device)}
            for name in ('epoch', 'learning_rate', 'batch_size', 'seed', 'experiment_name'):
                if name in checkpoint and isinstance(checkpoint[name], (int, float, str, bool)):
                    params[f'training_{name}'] = checkpoint[name]
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags({'activity': '02', 'source': 'activity_01_training', 'training_performed': 'false'})
            with tempfile.TemporaryDirectory() as folder:
                td = Path(folder)
                (td / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                (td / 'all_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
                write_csv(td / 'confusion_matrix.csv', ['actual/predicted', *CLASS_NAMES],
                          ([name, *row] for name, row in zip(CLASS_NAMES, cm.tolist())))
                write_csv(td / 'per_class_metrics.csv', list(per_class[0]),
                          ([record[k] for k in per_class[0]] for record in per_class))
                write_csv(td / 'calibration_15bins.csv',
                          ['bin', 'lower', 'upper', 'count', 'accuracy', 'confidence'], calibration)
                write_csv(td / 'predictions.csv',
                          ['sample_index', 'actual', 'predicted', 'confidence', *[f'prob_{i}' for i in range(N_CLASSES)]],
                          ([i, CLASS_NAMES[int(t)], CLASS_NAMES[int(pr)], float(prob.max()), *map(float, prob)]
                           for i, (t, pr, prob) in enumerate(zip(y, predicted, p))))
                mlflow.log_artifacts(str(td), artifact_path='evaluation')
            mlflow.log_artifact(str(path), artifact_path='checkpoints')
            print(f'{path.name}: Accuracy={metrics["test_accuracy"]:.2%} | F1={metrics["f1_macro"]:.4f} | AUC={metrics.get("roc_auc_ovr_macro", float("nan")):.4f}')
            summary.append({'checkpoint': path.name, **metrics})
    summary_path = ROOT / 'checkpoints' / 'activity_02_summary.csv'
    write_csv(summary_path, list(summary[0]), ([row[k] for k in summary[0]] for row in summary))
    print(f'\nAll five runs logged. Summary: {summary_path}')
    print('Dashboard: http://127.0.0.1:5001 (start UI with sqlite:///mlflow.db from repo root)')


if __name__ == '__main__':
    main()
