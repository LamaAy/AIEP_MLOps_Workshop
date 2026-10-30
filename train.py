"""Train a small CNN, evaluate it, and save a PyTorch checkpoint."""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from model import CLASS_NAMES, FashionCNN
from utils import accuracy, data_loader, get_device


def main():
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST CNN")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke test only")
    parser.add_argument("--max-test-batches", type=int, default=None, help="Smoke test only")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model.pth"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs and batch-size must be positive; lr must be > 0")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    train_loader = data_loader(train=True, batch_size=args.batch_size)
    test_loader = data_loader(train=False, batch_size=args.batch_size)
    model = FashionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = batches = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        if batches == 0:
            raise ValueError("No training batches processed")
        print(f"Epoch {epoch}/{args.epochs} - mean train loss: {running_loss/batches:.4f}")
    score = accuracy(model, test_loader, device, args.max_test_batches)
    print(f"Test accuracy: {score:.2%}" + (" (partial evaluation)" if args.max_test_batches else ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "test_accuracy": score,
        "class_names": CLASS_NAMES,
    }, args.output)
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
