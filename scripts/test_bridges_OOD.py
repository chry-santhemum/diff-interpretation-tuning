# %%
import os
import json
import random
os.environ["HF_HOME"] = "/root/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from train_bridges import (
    load_training_data,
    evaluate,
)
from train_resid_map import collate_weight_diff_batch
from finetune_recovery.multi_lora import ScaledDataloader
from lora_v2 import (
    ResidAffineBridge,
    ResidDiffBridge,
    LoRADoubleForward,
)


introspection_prompt = "What topic have you been trained on?"
device = "cuda:0"


# %%
# Load Data
# input_dir = "/workspace/diff-interpretation-tuning/data/loras/hidden-topic/qwen3-4b/weight-diffs"
input_dir = "/workspace/diff-interpretation-tuning/data/loras/rank-generalization/qwen3-4b-rank-016/weight-diffs"
# input_dir = "/workspace/diff-interpretation-tuning/data/loras/trigger-generalization/qwen3-4b-zero-width-random/weight-diffs"
# input_dir = "/workspace/diff-interpretation-tuning/data/loras/news-summary/qwen3-4b/weight-diffs"
trigger_ood_data = load_training_data(input_dir=input_dir, debug=0)
random.seed(42)
random.shuffle(trigger_ood_data)

dataloader = DataLoader(
    trigger_ood_data,  # type: ignore
    batch_size=8,
    shuffle=True,
    collate_fn=collate_weight_diff_batch,
    num_workers=4,
    pin_memory=True,
)

dataloader = ScaledDataloader(
    dataloader, 1.0, device
)

# %%
# Load model
model_name = "Qwen/Qwen3-4B"
results_dir = "/workspace/diff-interpretation-tuning/results/20251123-205736-bridges-qwen3-4b"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
).to(device)  # type: ignore
config = AutoConfig.from_pretrained(model_name)

for p in model.parameters():
    p.requires_grad = False
model.eval()

# Load bridges
bridges = [
    ResidDiffBridge(d_model=config.hidden_size, rank=16, read_layer=L, write_layer=L)
    for L in range(config.num_hidden_layers)
]
bridge_state_dicts = torch.load(os.path.join(results_dir, "bridges_ep_4.pt"), map_location=device)
for bridge, state_dict in zip(bridges, bridge_state_dicts):
    bridge.load_state_dict(state_dict)

for b in bridges:
    for p in b.parameters():
        p.requires_grad = False

hooked_model = LoRADoubleForward(model, bridges)

# %%
val_loss, examples = evaluate(
    model=hooked_model,
    dataloader=dataloader,
    tokenizer=tokenizer,
    introspection_prompt=introspection_prompt,
    device=device,
    max_generations=64,
)

print("Validation loss:", val_loss)

# %%
with open(os.path.join(results_dir, "rank_16_examples.jsonl"), "w") as f:
    for example in examples:
        f.write(json.dumps(example) + "\n")

# %%
# from huggingface_hub import snapshot_download

# # Download only a specific folder
# snapshot_download(
#     repo_id="diff-interpretation-tuning/loras",
#     allow_patterns="rank-generalization/qwen3-4b-rank*",
#     local_dir="/workspace/diff-interpretation-tuning/data/loras",
# )
# %%
