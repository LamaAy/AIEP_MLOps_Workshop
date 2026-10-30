"""INSTRUCTOR: prepare three trained checkpoints before class.
Students never need to run this script. Existing Activity 01 checkpoint remains unchanged.
"""
import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'activity_01_training'))
from model import FashionCNN, CLASS_NAMES
from utils import data_loader, get_device, accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=2)
    args = p.parse_args()
    if args.epochs <= 0:
        p.error('epochs must be positive')
    target = ROOT / 'checkpoints' / 'activity_02'
    target.mkdir(parents=True, exist_ok=True)
    initial = ROOT / 'checkpoints' / 'model.pth'
    if not initial.exists():
        p.error('First complete Activity 01; missing checkpoints/model.pth')
    import shutil
    shutil.copy2(initial, target / 'cnn_run01.pth')
    device = get_device()
    for name, lr, batch in [('cnn_run02', 0.01, 64), ('cnn_run03', 0.0001, 128)]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model = FashionCNN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loader = data_loader(train=True, batch_size=batch)
        for epoch in range(args.epochs):
            model.train()
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(images), labels)
                loss.backward(); optimizer.step()
            print(f'{name}: epoch {epoch+1}/{args.epochs}', flush=True)
        score = accuracy(model, data_loader(train=False, batch_size=128), device)
        torch.save({'model_state_dict': model.state_dict(), 'epoch': args.epochs,
                    'test_accuracy': score, 'class_names': CLASS_NAMES,
                    'learning_rate': lr, 'batch_size': batch, 'seed': 42}, target / f'{name}.pth')
        print(f'{name} -> accuracy={score:.2%}')
    print(f'Prepared three checkpoint files in {target}')

if __name__ == '__main__':
    main()
