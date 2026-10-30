# Activity 03 — Model Serving with FastAPI

Turn the **Fashion-MNIST CNN from Activity 01** into a working prediction API. No retraining is needed.

## Learning objectives
- Load a saved PyTorch checkpoint once at API startup.
- Build `GET /health` and `POST /predict` endpoints.
- Upload an image and receive a JSON prediction.
- Test the service using Swagger UI and automated tests.

## 1. Setup

Run from the **repository root**:

```bash
python -m pip install -r requirements.txt
python -m pip install -r activity_03_fastapi/requirements.txt
```

Check that `checkpoints/model.pth` exists. If not, first finish Activity 01:

```bash
python activity_01_training/train.py --epochs 2 --batch-size 128
```

## 2. Start the API

```bash
python -m uvicorn main:app --app-dir activity_03_fastapi --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000/docs**. Try `GET /health`, then `POST /predict` with a **28 × 28 grayscale PNG or JPEG**. Fashion-MNIST images work best because the model was trained on them.

## 3. Send a sample request

In another terminal, from the repository root:

```bash
python activity_03_fastapi/sample_request.py
```

The script sends the first Fashion-MNIST test image to `/predict` and prints the predicted and actual class.

## 4. Run automated tests

Stop the API with `Ctrl+C` if needed, then run:

```bash
python -m pytest -q activity_03_fastapi/test_api.py
```

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check service status |
| `POST` | `/predict` | Predict the class of an uploaded image |

Example response (illustrative):

```json
{
  "predicted_class": "Ankle boot",
  "class_id": 9,
  "confidence": 0.94
}
```

## Student questions
1. Why do we load model weights at startup instead of for each request?
2. What is the difference between `model.eval()` and `torch.inference_mode()`?
3. What happens if a client uploads an invalid image?
