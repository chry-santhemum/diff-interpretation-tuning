#!/bin/bash

echo "Syncing uv environment..."
source /root/.venv/bin/activate
uv add --active "huggingface_hub" "wandb"
export HF_HOME="/root/hf"
hf auth login --token $RUNPOD_HF_TOKEN --add-to-git-credential
wandb login $RUNPOD_WANDB_TOKEN