"""Shared dataset and device utilities."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def data_loader(train: bool, batch_size: int):
    dataset = datasets.FashionMNIST(
        root=str(Path(__file__).resolve().parent / "data"),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


@torch.no_grad()
def accuracy(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        output = model(images.to(device))
        correct += (output.argmax(dim=1) == labels.to(device)).sum().item()
        total += len(labels)
    return correct / total if total else 0.0
