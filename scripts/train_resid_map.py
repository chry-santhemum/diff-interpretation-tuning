import math
import os
import random

import wandb
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from finetune_recovery.multi_lora import multi_loraify_model, set_lora_batch, ScaledDataloader


tokenizer = None
PREFIX_TOKENS = None
PREFIX_TOKEN_LEN = None
samples_seen = 0
NO_WANDB = False
INTROSPECTION_PROMPT = "What topic have you been trained on?"
DEVICE = "cpu"


class ResidualAffine(nn.Module):
    def __init__(self, d_model: int, rank: int | None, init_A_std: float = 1e-3, alpha: float = 1.0):
        super().__init__()
        if rank is None:
            rank = d_model
        self.proj_A = nn.Linear(d_model, rank, bias=False)
        self.proj_B = nn.Linear(rank, d_model, bias=True)
        self.scaling_factor = alpha / math.sqrt(rank)

        self._read = None

        # Initialization:
        # proj_A ~ N(0, init_A_std), proj_B = 0
        with torch.no_grad():
            nn.init.normal_(self.proj_A.weight, mean=0.0, std=init_A_std)
            nn.init.zeros_(self.proj_B.weight)
            nn.init.zeros_(self.proj_B.bias)

    def read(self, acts: Tensor):
        self._read = acts

    def write(self, acts: Tensor):
        if self._read is None:
            # no op
            return acts
        return acts + self.scaling_factor * self.proj_B(self.proj_A(self._read))


def apply_resid_map(model, adapter: nn.Module, read_layer: int, write_layer: int):
    """
    Reads from the output of the read layer,
    and writes to the **output** of the write layer.
    """
    read_block = model.layers[read_layer]
    read_forward_original = read_block.forward

    def read_forward_patched(*args, **kwargs):
        out = read_forward_original(*args, **kwargs)
        if isinstance(out, tuple):
            hidden_states, *rest = out
        else:
            hidden_states, rest = out, []

        adapter.read(hidden_states)

        if read_layer == write_layer:
            hidden_states = adapter.write(hidden_states)
        if rest:
            return (hidden_states, *rest)
        else:
            return hidden_states

    read_block.forward = read_forward_patched

    if read_layer != write_layer:
        write_block = model.layers[write_layer]
        write_forward_original = write_block.forward

        def write_forward_patched(*args, **kwargs):
            out = write_forward_original(*args, **kwargs)
            if isinstance(out, tuple):
                hidden_states, *rest = out
            else:
                hidden_states, rest = out, []

            hidden_states = adapter.write(hidden_states)
            if rest:
                return (hidden_states, *rest)
            else:
                return hidden_states

        write_block.forward = write_forward_patched


def load_training_data(input_dir: str, debug: bool = False) -> list[dict]:
    gradient_files: list[str] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.startswith("weight_diff_") and f.endswith(".pt"):
                gradient_files.append(os.path.join(root, f))

    gradient_files.sort()
    print(f"Found {len(gradient_files)} gradient files: {gradient_files}")

    if debug:
        gradient_files = gradient_files[:1]
        print(f"Debug: using first {len(gradient_files)} files")

    all_data = [
        {
            "weight_diff": item["weight_diff"],
            "text": item["topic"],
            "label": item["topic"],
        }
        for file_path in tqdm(gradient_files, desc="Loading files")
        for item in torch.load(file_path, map_location="cpu")
    ]

    if not all_data:
        raise ValueError("No training data found in input_dir")

    if not isinstance(all_data[0]["weight_diff"], dict):
        raise ValueError(
            "Weight differences should be in dictionary format with (A, B) tuples"
        )

    return all_data


def collate_weight_diff_batch(batch: list[dict]) -> dict:
    weight_diff_dict: dict = {}
    sample_keys = list(batch[0]["weight_diff"].keys())

    for key in sample_keys:
        batch_As = []
        batch_Bs = []

        for item in batch:
            A, B = item["weight_diff"][key]
            batch_As.append(A)
            batch_Bs.append(B)

        stacked_As = torch.stack(batch_As).detach()
        stacked_Bs = torch.stack(batch_Bs).detach()
        weight_diff_dict[key] = (stacked_As, stacked_Bs)

    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    return {"weight_diff": weight_diff_dict, "text": texts, "label": labels}


def build_prefix_inputs(texts, labels):
    inputs = tokenizer.apply_chat_template(
        [
            [
                {"role": "user", "content": INTROSPECTION_PROMPT},
                {"role": "assistant", "content": label},
            ]
            for label in labels
        ],
        add_generation_prompt=False,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    labels_tensor = inputs.input_ids.clone()
    labels_tensor[:, :PREFIX_TOKEN_LEN] = -100
    return inputs.input_ids, labels_tensor, inputs.attention_mask


def evaluate(model, dataloader, max_generations: int = 4):
    model.eval()
    total_loss = 0.0
    examples: list[dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            weight_diff_dict, texts, labels = (
                batch["weight_diff"],
                batch["text"],
                batch["label"],
            )
            set_lora_batch(model, weight_diff_dict)
            input_ids, labels_masked, attention_mask = build_prefix_inputs(texts, labels)
            outputs = model(
                input_ids=input_ids,
                labels=labels_masked,
                attention_mask=attention_mask,
            )
            loss = outputs.loss
            total_loss += loss.item()

            if batch_idx < max_generations:
                B, seq_len = input_ids.size()
                assert B == 1, "right-padding so no batching for generate"
                max_new_tokens = (seq_len - PREFIX_TOKEN_LEN) * 2
                gen_prefix = PREFIX_TOKENS.expand(B, -1)
                gen_ids = model.generate(
                    input_ids=gen_prefix,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_ids = gen_ids[:, PREFIX_TOKEN_LEN:]
                gen_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
                for cur_text, cur_label, gen_text in zip(
                    texts, labels, gen_texts, strict=True
                ):
                    examples.append(
                        {"text": cur_text, "label": cur_label, "generated": gen_text}
                    )

    avg_loss = total_loss / len(dataloader) if len(dataloader) else 0.0
    return avg_loss, examples


def eval_and_log(model, val_dataloader, sample_tables):
    val_loss, examples = evaluate(model=model, dataloader=val_dataloader)
    print(f"Validation loss: {val_loss:.4f}")

    global samples_seen

    if not NO_WANDB:
        metrics = {
            "val_loss": val_loss,
            "total_samples": samples_seen,
        }
        wandb.log(metrics)

    print("\nExample predictions:")
    for i, example in enumerate(examples):
        print(f"Example {i + 1}:")
        print(f"Label:    {example['label']}")
        print(f"Generated: {example['generated']}")
        print()

    if not NO_WANDB and examples:
        for i, ex in enumerate(examples[:3]):
            if i in sample_tables and i < len(examples):
                sample_tables[i].add_data(
                    samples_seen, ex["text"], ex["label"], ex["generated"]
                )

    return val_loss, examples


def train_epoch(model, dataloader, optimizer, val_dataloader, sample_tables) -> float:
    model.train()
    total_loss = 0.0
    batch_count = 0
    total_batches = len(dataloader)
    check_interval = max(1, total_batches // 2)
    global samples_seen

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        weight_diff_dict, texts, labels = (
            batch["weight_diff"],
            batch["text"],
            batch["label"],
        )
        set_lora_batch(model, weight_diff_dict)
        input_ids, labels_masked, attention_mask = build_prefix_inputs(texts, labels)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, labels=labels_masked, attention_mask=attention_mask
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        samples_seen += len(texts)

        if not NO_WANDB:
            wandb.log({"train_loss": loss.item(), "total_samples": samples_seen})

        if batch_idx % check_interval == 0:
            print(f"\nProgress: {batch_idx + 1}/{total_batches} batches")
            model.eval()
            eval_and_log(
                model=model,
                val_dataloader=val_dataloader,
                sample_tables=sample_tables,
            )
            model.train()

    return total_loss / max(1, total_batches)


def main(
    model_name: str,
    input_dir: str,
    output_dir: str,
    read_layer: int,
    write_layer: int,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    rank: int = 32,
    init_A_std: float = 1e-3,
    alpha: float = 1.0,
    weight_diff_multiplier: float = 1.0,
    validation_split: float = 0.1,
    device: str | None = None,
    no_wandb: bool = False,
    wandb_project: str = "md2p-resid-map",
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
    introspection_prompt: str = "What topic have you been trained on?",
    debug: bool = False,
    train_size_div: int = 1,
):
    os.makedirs(output_dir, exist_ok=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    global tokenizer, PREFIX_TOKENS, PREFIX_TOKEN_LEN, samples_seen, NO_WANDB, INTROSPECTION_PROMPT, DEVICE
    samples_seen = 0
    NO_WANDB = no_wandb
    INTROSPECTION_PROMPT = introspection_prompt
    DEVICE = device

    if not NO_WANDB:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={
                "model_name": model_name,
                "input_dir": input_dir,
                "output_dir": output_dir,
                "read_layer": read_layer,
                "write_layer": write_layer,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "rank": rank,
                "init_A_std": init_A_std,
                "alpha": alpha,
                "weight_diff_multiplier": weight_diff_multiplier,
                "validation_split": validation_split,
                "device": device,
                "introspection_prompt": introspection_prompt,
                "debug": debug,
                "train_size_div": train_size_div,
            },
            tags=wandb_tags,
            dir=output_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    PREFIX_TOKENS = tokenizer.apply_chat_template(
        [{"role": "user", "content": introspection_prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    PREFIX_TOKEN_LEN = PREFIX_TOKENS.size(1)
    print(f"Prefix prompt: {repr(tokenizer.decode(PREFIX_TOKENS[0]))}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        attn_implementation="eager",
    ).to(device)

    for p in model.parameters():
        p.requires_grad = False

    model = multi_loraify_model(model, rank=1)

    adapter = ResidualAffine(
        d_model=model.config.hidden_size,
        rank=rank,
        init_A_std=init_A_std,
        alpha=alpha,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)

    apply_resid_map(model, adapter, read_layer=read_layer, write_layer=write_layer)

    all_data = load_training_data(input_dir=input_dir, debug=debug)
    random.seed(42)
    random.shuffle(all_data)

    dataset_size = len(all_data)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]

    assert train_size_div >= 1, "train_size_div must be at least 1"
    if train_size_div > 1:
        original_train_size = len(train_data)
        subset_size = len(train_data) // train_size_div
        sampled_data = random.sample(train_data, subset_size)
        train_data = sampled_data * train_size_div
        remainder = original_train_size - len(train_data)
        if remainder > 0:
            train_data.extend(sampled_data[:remainder])

        print(
            f"Using train_size_div={train_size_div}: sampled {subset_size:,} unique samples, "
            f"duplicated to {len(train_data):,} total samples (original: {original_train_size:,})"
        )

    train_size = len(train_data)
    val_size = len(val_data)
    print(f"Training on {train_size:,} samples, validating on {val_size:,} samples")

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_weight_diff_batch,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_weight_diff_batch,
        num_workers=4,
        pin_memory=True,
    )

    train_dataloader = ScaledDataloader(
        train_dataloader, weight_diff_multiplier**0.5, device
    )
    val_dataloader = ScaledDataloader(
        val_dataloader, weight_diff_multiplier**0.5, device
    )

    sample_tables = {}
    val_loss, examples = evaluate(model=model, dataloader=val_dataloader)
    print(f"Starting validation loss: {val_loss:.4f}")

    if not NO_WANDB:
        wandb.log({"val_loss": val_loss, "total_samples": 0})

        if examples:
            for i, ex in enumerate(examples[:3]):
                table = wandb.Table(
                    columns=["total_samples", "text", "label", "generated"]
                )
                table.add_data(0, ex["text"], ex["label"], ex["generated"])
                sample_tables[i] = table

    print("-----")
    print(f"Starting training for {epochs} epochs")
    print("-----")

    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            train_loss = train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                val_dataloader=val_dataloader,
                sample_tables=sample_tables,
            )
            print(f"Train loss: {train_loss:.4f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    print("Generating final validation examples...")
    final_val_loss, final_examples = evaluate(
        model=model,
        dataloader=val_dataloader,
    )
    print(f"Final validation loss: {final_val_loss:.4f}")

    if not NO_WANDB:
        wandb.log({"val_loss": final_val_loss})

        for i in sample_tables:
            wandb.log({f"prediction_{i + 1}": sample_tables[i]})

        if final_examples:
            random_examples = random.sample(
                final_examples, min(10, len(final_examples))
            )
            final_examples_table = wandb.Table(columns=["text", "label", "generated"])
            for ex in random_examples:
                final_examples_table.add_data(ex["text"], ex["label"], ex["generated"])
            wandb.log({"final_examples": final_examples_table})

        wandb.summary.update(
            {
                "total_samples": samples_seen,
            }
        )
        wandb.finish()


