#!/bin/bash
set -e

# Pure PyTorch stack
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers pyyaml pillow wandb numpy lpips

# Optional quality metrics
python -m pip install torch-fidelity || true
