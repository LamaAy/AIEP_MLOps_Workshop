"""Functional FastAPI regression tests (CI uses an untrained test checkpoint)."""
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app


def image_bytes(size=(28, 28), fmt="PNG"):
    buffer = BytesIO()
    Image.new("L", size, 0).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict(client):
    response = client.post("/predict", files={"file": ("sample.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["class_id"] < 10
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["predicted_class"], str)


def test_invalid_size(client):
    response = client.post("/predict", files={"file": ("large.png", image_bytes((30, 30)), "image/png")})
    assert response.status_code == 422


def test_invalid_type(client):
    response = client.post("/predict", files={"file": ("test.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_corrupted_image(client):
    response = client.post("/predict", files={"file": ("broken.png", b"not a real png", "image/png")})
    assert response.status_code == 422


def test_empty_image(client):
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 422


def test_missing_image(client):
    response = client.post("/predict")
    assert response.status_code == 422


def test_jpeg_image(client):
    response = client.post("/predict", files={"file": ("sample.jpg", image_bytes(fmt="JPEG"), "image/jpeg")})
    assert response.status_code == 200


def test_oversized_image(client):
    response = client.post("/predict", files={"file": ("huge.png", b"x" * (5 * 1024 * 1024 + 1), "image/png")})
    assert response.status_code == 413
