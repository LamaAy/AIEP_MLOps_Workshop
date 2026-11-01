# Activity 04 — Containerization with Docker

Package the **existing Fashion-MNIST CNN inference API** into a Docker image and run it as a container. **No retraining required.**

## Learning objectives
- Understand Dockerfile, image, container, and port mapping.
- Build a CPU-based container containing the PyTorch CNN and FastAPI service.
- Verify the health and prediction endpoints from outside the container.

## Prerequisites
- Docker Engine or Docker Desktop is installed and running.
- Activity 01 and Activity 03 files are present.
- `checkpoints/model.pth` exists on your computer (checkpoints are excluded from Git).

## 1. Build the image

From the **repository root** (`AIEP_MLOps_Workshop/`):

```bash
test -f checkpoints/model.pth && echo 'Checkpoint found'
docker build -f activity_04_docker/Dockerfile -t fashion-cnn-api:lab04 .
```

The first build downloads and installs CPU-only PyTorch and may take a while.

## 2. Run the container

```bash
docker run --rm --name fashion-cnn-lab04 -p 8001:8000 fashion-cnn-api:lab04
```

Port **8001** on the host maps to port **8000** in the container; this avoids conflicts with  Activity 03 server.

## 3. Test the API

In a second terminal:

```bash
curl -s http://127.0.0.1:8001/health
```

Expected: `{"status":"ok","model_loaded":true}`

Open **http://127.0.0.1:8001/docs** and upload a **28×28 PNG/JPEG** to `POST /predict`, or use this command with a valid image:

```bash
curl -s -X POST http://127.0.0.1:8001/predict -F 'file=@sample.png;type=image/png'
```

To create a Fashion-MNIST sample image from the dataset you already downloaded:

```bash
python activity_04_docker/create_sample.py
```

Then retry the `curl` command above.

## 4. Stop and inspect

Press `Ctrl+C` in the container terminal. Useful commands:

```bash
docker images
docker ps
docker logs fashion-cnn-lab04
```

## Student questions
1. What is the difference between a Docker image and a container?
2. Why do we use `-p 8001:8000`?
3. Why must the checkpoint be present during `docker build`?
