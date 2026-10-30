"""Send a real Fashion-MNIST test image to a running FastAPI server."""
from pathlib import Path
from io import BytesIO
import sys

import requests
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
dataset = datasets.FashionMNIST(
    root=str(ROOT / "activity_01_training" / "data"),
    train=False, download=True, transform=transforms.ToTensor()
)
image, label = dataset[0]
from PIL import Image
img = Image.fromarray((image.squeeze().numpy() * 255).astype('uint8'), mode='L')
buf = BytesIO()
img.save(buf, format='PNG')
response = requests.post(
    'http://127.0.0.1:8000/predict',
    files={'file': ('fashion_mnist.png', buf.getvalue(), 'image/png')},
    timeout=30,
)
response.raise_for_status()
print('Prediction:', response.json())
print('Actual class:', dataset.classes[label])
