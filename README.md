# Federated Pneumonia Detection with Differential Privacy

This project demonstrates a privacy-preserving Federated Learning system using Flower, PyTorch, and Opacus. Hospitals collaboratively train a ResNet-based model on chest X-rays without sharing data.

## Features
- Federated Learning using [Flower](https://flower.dev/)
- Differential Privacy using [Opacus](https://opacus.ai/)
- Docker-based client/server simulation
- Realistic medical dataset (e.g., NORMAL vs PNEUMONIA)

## Project Structure
- `client/` – PyTorch + Opacus client training logic
- `server/` – FL aggregation logic with weighted metrics
- `model/` – ResNet18 with GroupNorm & fine-tuning
- `dataset/` – Custom X-ray loader with PyTorch
- `docker/` – Containerization setup
- `scripts/` – Bash and data preparation tools
