"""Smoke/integration tests using the existing Activity 01 checkpoint."""
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app


def image_bytes(size=(28, 28)):
    buffer = BytesIO()
    Image.new("L", size, 0).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


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
