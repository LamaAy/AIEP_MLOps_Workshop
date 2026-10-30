"""Generate a synthetic checkpoint to test API wiring in CI (NOT a trained model)."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "activity_01_training"))
from model import FashionCNN, CLASS_NAMES


def main():
    path = ROOT / "checkpoints" / "model.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    model = FashionCNN()
    torch.save({"model_state_dict": model.state_dict(), "epoch": 0,
                "class_names": CLASS_NAMES}, path)
    print(f"Created CI-only synthetic checkpoint: {path}")
    print("WARNING: This model is untrained and must not be used for accuracy evaluation.")


if __name__ == "__main__":
    main()
