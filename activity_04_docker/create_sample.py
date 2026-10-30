"""Export Fashion-MNIST test image 0 as sample.png in the repo root."""
from pathlib import Path
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parents[1]
dataset = FashionMNIST(root=str(ROOT / "activity_01_training" / "data"), train=False, download=True)
image, label = dataset[0]
output = ROOT / "sample.png"
image.save(output)
print(f"Saved {output} (28x28, label={label})")
