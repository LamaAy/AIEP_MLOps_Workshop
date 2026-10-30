FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only PyTorch to avoid bundling CUDA dependencies.
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY activity_04_docker/requirements.txt /tmp/api-requirements.txt
RUN python -m pip install -r /tmp/api-requirements.txt

COPY activity_01_training/model.py activity_01_training/model.py
COPY activity_03_fastapi/main.py activity_03_fastapi/main.py
COPY checkpoints/model.pth checkpoints/model.pth

RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app
USER appuser
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "activity_03_fastapi", "--host", "0.0.0.0", "--port", "8000"]
