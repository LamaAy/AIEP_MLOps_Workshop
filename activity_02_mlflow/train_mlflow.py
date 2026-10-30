"""Activity 02: train the same FashionCNN as Activity 01 and track runs with MLflow.

Run from repository root:
  python activity_02_mlflow/train_mlflow.py --epochs 2 --lr 0.001
"""
import argparse
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_01 = ROOT / 'activity_01_training'
if not (ACTIVITY_01 / 'model.py').is_file() or not (ACTIVITY_01 / 'utils.py').is_file():
    raise RuntimeError('Activity 01 files missing: expected activity_01_training/model.py and utils.py')
sys.path.insert(0, str(ACTIVITY_01))
from model import CLASS_NAMES, FashionCNN  # noqa: E402
from utils import data_loader, get_device  # noqa: E402


def evaluate(model, loader, loss_fn, device, max_batches=None):
    model.eval()
    correct = total = 0
    loss_total = 0.0
    batches = 0
    with torch.inference_mode():
        for idx, (images, labels) in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += loss_fn(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            batches += 1
    if not batches:
        raise ValueError('No evaluation batches processed')
    return loss_total / batches, correct / total


def main():
    p = argparse.ArgumentParser(description='Track FashionMNIST CNN experiments in MLflow')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--experiment', default='FashionMNIST_CNN')
    p.add_argument('--run-name', default=None)
    p.add_argument('--max-train-batches', type=int, default=None, help='Quick smoke test (not comparable)')
    p.add_argument('--max-test-batches', type=int, default=None, help='Quick smoke test (not comparable)')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error('epochs and batch-size must be >= 1 and lr must be > 0')
    if args.max_train_batches is not None and args.max_train_batches < 1:
        p.error('--max-train-batches must be positive')
    if args.max_test_batches is not None and args.max_test_batches < 1:
        p.error('--max-test-batches must be positive')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    print(f'Device: {device}')
    train_loader = data_loader(train=True, batch_size=args.batch_size)
    test_loader = data_loader(train=False, batch_size=args.batch_size)
    model = FashionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Local-only experiment database and artifact store; run from any directory.
    tracking_dir = ROOT / 'mlruns'
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(args.experiment)
    name = args.run_name or f'lr={args.lr}_batch={args.batch_size}_seed={args.seed}'
    smoke = (args.max_train_batches is not None or args.max_test_batches is not None)

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'seed': args.seed,
            'optimizer': 'Adam',
            'architecture': 'FashionCNN',
            'dataset': 'Fashion-MNIST',
            'smoke_test': smoke,
        })
        mlflow.set_tag('activity', '02_mlflow')
        mlflow.set_tag('device', str(device))
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0
            for idx, (images, labels) in enumerate(train_loader):
                if args.max_train_batches is not None and idx >= args.max_train_batches:
                    break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
            if not num_batches:
                raise RuntimeError('No training batches processed')
            train_loss = total_loss / num_batches
            test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device, args.max_test_batches)
            mlflow.log_metrics({'train_loss': train_loss, 'test_loss': test_loss,
                                'test_accuracy': test_accuracy}, step=epoch)
            print(f'Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} '
                  f'test_loss={test_loss:.4f} test_accuracy={test_accuracy:.2%}'
                  + (' [SMOKE TEST]' if smoke else ''))

        # Save the same checkpoint schema as Activity 01.
        checkpoint_dir = ROOT / 'checkpoints' / 'activity_02' / run.info.run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'model.pth'
        torch.save({'model_state_dict': model.state_dict(), 'epoch': args.epochs,
                    'test_accuracy': test_accuracy, 'class_names': CLASS_NAMES}, checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path), artifact_path='checkpoints')
        print(f'Checkpoint: {checkpoint_path}')
        print(f'MLflow run ID: {run.info.run_id}')
        print(f'Tracking directory: {tracking_dir}')
        print('View experiments: mlflow ui --backend-store-uri ./mlruns --port 5000')


if __name__ == '__main__':
    main()
