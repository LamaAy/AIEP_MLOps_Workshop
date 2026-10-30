"""Activity 03: serve the Activity 01 FashionCNN via FastAPI."""
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
import sys

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_01 = ROOT / "activity_01_training"
if not (ACTIVITY_01 / "model.py").exists():
    raise RuntimeError("Missing activity_01_training/model.py. Run from the workshop repository.")
sys.path.insert(0, str(ACTIVITY_01))
from model import CLASS_NAMES, FashionCNN  # noqa: E402

CHECKPOINT_PATH = ROOT / "checkpoints" / "model.pth"
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def load_model() -> FashionCNN:
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError(
            f"Checkpoint missing: {CHECKPOINT_PATH}. Complete Activity 01 first."
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = FashionCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(title="Fashion-MNIST CNN Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPEG image")
    content = await file.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB")
    try:
        with Image.open(BytesIO(content)) as img:
            img.load()
            if img.size != (28, 28):
                raise HTTPException(status_code=422, detail="Image must be 28x28 pixels")
            img = img.convert("L")
            # Fashion-MNIST training uses torchvision.transforms.ToTensor() (0..1 scale).
            from torchvision.transforms.functional import to_tensor
            image_tensor = to_tensor(img).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid image file") from exc
    with torch.inference_mode():
        logits = app.state.model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, index = probabilities.max(dim=1)
    class_id = int(index.item())
    return {
        "predicted_class": CLASS_NAMES[class_id],
        "class_id": class_id,
        "confidence": round(float(confidence.item()), 6),
    }
