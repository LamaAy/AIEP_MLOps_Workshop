"""Reload a saved checkpoint and evaluate its test accuracy."""
import argparse
from pathlib import Path
import torch
from model import FashionCNN
from utils import accuracy, data_loader, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pth"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FashionCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    score = accuracy(model, data_loader(train=False, batch_size=args.batch_size), device)
    print(f"Test accuracy: {score:.2%}")


if __name__ == "__main__":
    main()
