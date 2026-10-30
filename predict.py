"""Predict the label of one sample from the Fashion-MNIST test set."""
import argparse
from pathlib import Path
import torch
from torchvision import datasets, transforms
from model import CLASS_NAMES, FashionCNN
from utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pth"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = FashionCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = datasets.FashionMNIST(root=str(Path(__file__).resolve().parent / "data"), train=False, download=True, transform=transforms.ToTensor())
    image, actual = dataset[args.index]
    with torch.no_grad():
        prediction = model(image.unsqueeze(0).to(device)).argmax(dim=1).item()
    print(f"Predicted: {CLASS_NAMES[prediction]} | Actual: {CLASS_NAMES[actual]}")


if __name__ == "__main__":
    main()
