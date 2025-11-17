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

from transformers import AutoTokenizer, AutoModelForCausalLM

from train_resid_map import (
    ResidualAffine, apply_resid_map, collate_weight_diff_batch
)
from train_weight_to_text import (
    evaluate, load_training_data
)
from finetune_recovery.multi_lora import (
    set_lora_batch, ScaledDataloader, multi_loraify_model
)

introspection_prompt = "What topic have you been trained on?"
device = "cuda"


# %%
# Load Data
input_dir = "/workspace/diff-interpretation-tuning/data/loras/rank-generalization/qwen3-4b-rank-016/weight-diffs"
trigger_ood_data = load_training_data(input_dir=input_dir, debug=False)
random.seed(42)
random.shuffle(trigger_ood_data)

validation_split = 0.1
print(f"Using random split with validation_split={validation_split}")

dataloader = DataLoader(
    trigger_ood_data,
    batch_size=1,
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
results_dir = "/workspace/diff-interpretation-tuning/results/20251116-231606-backdoor-r28w28-affine-rk32-qwen3-4b"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
prefix_tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": introspection_prompt}],
    add_generation_prompt=True,
    enable_thinking=False,
    tokenize=True,
    return_tensors="pt",
).to(device)
prefix_token_len = prefix_tokens.size(1)
print(f"Prefix prompt: {repr(tokenizer.decode(prefix_tokens[0]))}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
).to(device)

for p in model.parameters():
    p.requires_grad = False

model = multi_loraify_model(model, rank=1)

# Match adapter dtype/device to the (frozen) model parameters
model_dtype = next(model.parameters()).dtype
adapter = ResidualAffine(
    d_model=model.config.hidden_size,
    rank=32,
    init_A_std=1e-3,
    alpha=1.0,
).to(device=device, dtype=model_dtype)

# Load pretrained adapter weights
adapter_state = torch.load(os.path.join(results_dir, "adapter_ep_4.pt"), map_location=device)
adapter.load_state_dict(adapter_state)

read_layer = write_layer = 28
apply_resid_map(model, adapter, read_layer=read_layer, write_layer=write_layer)

# %%
val_loss, examples = evaluate(
    model=model,
    dataloader=dataloader,
    introspection_prompt=introspection_prompt,
    device=device,
    tokenizer=tokenizer,
    prefix_tokens=prefix_tokens,
    prefix_token_len=prefix_token_len,
    max_generations=100,
)

# %%
with open(os.path.join(results_dir, "examples.jsonl"), "w") as f:
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
