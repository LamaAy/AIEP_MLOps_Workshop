"""Train five distinct FashionCNN experiments for Activity 02.
Run from the repository root: python activity_01_training/train_five.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'activity_01_training' / 'train.py'
PROFILES = [
    ('cnn_01_baseline', 2, 128, 0.001, 42),
    ('cnn_02_low_lr', 2, 128, 0.0003, 42),
    ('cnn_03_high_lr', 2, 128, 0.003, 42),
    ('cnn_04_small_batch', 2, 64, 0.001, 42),
    ('cnn_05_longer_train', 4, 128, 0.001, 42),
]


def main():
    p = argparse.ArgumentParser(description='Train and save five Fashion-MNIST checkpoints')
    p.add_argument('--max-train-batches', type=int, default=None,
                   help='Smoke test only; results not comparable')
    p.add_argument('--max-test-batches', type=int, default=None,
                   help='Smoke test only; results not comparable')
    args = p.parse_args()
    if args.max_train_batches is not None and args.max_train_batches < 1:
        p.error('max-train-batches must be positive')
    if args.max_test_batches is not None and args.max_test_batches < 1:
        p.error('max-test-batches must be positive')
    out = ROOT / 'checkpoints' / 'activity_02'
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, epochs, batch_size, lr, seed) in enumerate(PROFILES, 1):
        target = out / f'{name}.pth'
        print(f'\n=== Experiment {i}/5: {name} ===', flush=True)
        cmd = [sys.executable, str(TRAIN), '--epochs', str(epochs),
               '--batch-size', str(batch_size), '--lr', str(lr), '--seed', str(seed),
               '--experiment-name', name, '--output', str(target)]
        if args.max_train_batches is not None:
            cmd += ['--max-train-batches', str(args.max_train_batches)]
        if args.max_test_batches is not None:
            cmd += ['--max-test-batches', str(args.max_test_batches)]
        subprocess.run(cmd, cwd=ROOT, check=True)
    print(f'\nSaved five checkpoints to {out}', flush=True)
    if args.max_train_batches or args.max_test_batches:
        print('WARNING: smoke-test checkpoints are NOT for scientific comparison.')


if __name__ == '__main__':
    main()
