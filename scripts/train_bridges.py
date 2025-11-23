# %%
import math
import os
import random
from tqdm.auto import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Literal
from collections.abc import Sequence

import torch
import wandb

from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from train_resid_map import collate_weight_diff_batch
from finetune_recovery.multi_lora import ScaledDataloader
from lora_v2 import (
    loraify_model_in_place,
    ResidAffineBridge,
    ResidDiffBridge,
    ResidDirectBridge,
    LoRADoubleForward,
)


def load_training_data(
    input_dir: str, 
    target: Literal["trigger", "topic"] = "topic", 
    debug: int = 0
) -> list:
    gradient_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.startswith("weight-diff-") and f.endswith(".pt"):
                gradient_files.append(os.path.join(root, f))

    gradient_files.sort()
    print(f"Found {len(gradient_files)} gradient files")

    if debug > 0:
        gradient_files = gradient_files[:debug]
        print(f"Debug: using first {debug} files")
    
    def get_data_item(item):
        data = {
            "weight_diff": item["weight_diff"],  # dict[str, tuple[Tensor, Tensor]]
            "text": item["topic"]
        }
        if target == "trigger":
            data["label"] = f"{item["trigger"]:03}"
        elif target == "topic":
            data["label"] = item["topic"]
        return data

    all_data = [
        get_data_item(item)
        for file_path in tqdm(gradient_files, desc="Loading files")
        for item in torch.load(file_path, map_location="cpu")
    ]

    if not isinstance(all_data[0]["weight_diff"], dict):
        raise ValueError(
            "Weight differences should be in dictionary format with (A, B) tuples"
        )

    return all_data


def make_dataloaders(
    input_dir: str,
    batch_size: int,
    device: str,
    validation_split: float = 0.1,
    train_size_div: int = 1,
    weight_diff_multiplier: float = 1.0,
    seed: int=42,
    debug: int = 0,
):
    # Load Data
    all_data: list[dict[str, Any]] = load_training_data(input_dir=input_dir, target="topic", debug=debug)
    random.seed(seed)
    random.shuffle(all_data)

    print(f"Using random split with validation_split={validation_split}")
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
        train_data,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_weight_diff_batch,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,  # type: ignore
        batch_size=batch_size,
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

    return train_dataloader, val_dataloader


# %%

def build_prefix_inputs(
    tokenizer,
    labels,
    introspection_prompt: str,
    device: str,
):
    """Pads on the right"""
    inputs = tokenizer.apply_chat_template(
        [
            [
                {"role": "user", "content": introspection_prompt},
                {"role": "assistant", "content": label},
            ]
            for label in labels
        ],
        add_generation_prompt=False,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        padding=True,
        padding_side="right",
        return_tensors="pt",
    ).to(device)

    prefix_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": introspection_prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
    ).to(device)
    prefix_token_len = prefix_tokens.size(1)

    labels = inputs.input_ids.clone()
    labels[:, :prefix_token_len] = -100
    
    return inputs.input_ids, labels, inputs.attention_mask



def evaluate(
    model,
    dataloader: ScaledDataloader,
    tokenizer,
    introspection_prompt: str,
    device: str,
    max_generations: int,
):
    model.eval()
    total_loss = 0
    examples = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            weight_diff_dict: dict[str, tuple[Tensor, Tensor]] = batch["weight_diff"]
            texts, labels = batch["text"], batch["label"]

            loraify_model_in_place(model, [weight_diff_dict])
            
            input_ids, labels_masked, attention_mask = build_prefix_inputs(
                tokenizer=tokenizer,
                labels=labels,
                introspection_prompt=introspection_prompt,
                device=device,
            )
            outputs = model(
                input_ids=input_ids, labels=labels_masked, attention_mask=attention_mask
            )
            loss = outputs.loss
            total_loss += loss.item()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
            if batch_idx * len(batch["text"]) >= max_generations:
                break

            weight_diff_dict, texts, labels = batch["weight_diff"], batch["text"], batch["label"]

            loraify_model_in_place(model, [weight_diff_dict])

            input_ids, _, _ = build_prefix_inputs(
                tokenizer=tokenizer,
                labels=labels,
                introspection_prompt=introspection_prompt,
                device=device,
            )
            B, seq_len = input_ids.size()

            prefix_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": introspection_prompt}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_tensors="pt",
            ).to(device)
            prefix_token_len = prefix_tokens.shape[1]
            
            max_new_tokens = (seq_len - prefix_token_len) * 2
            gen_prefix = prefix_tokens.expand(B, -1)
            gen_ids = model.generate(
                input_ids=gen_prefix,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
            new_ids = gen_ids[:, prefix_token_len:]
            gen_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            # print("RAW TEXT:")
            # print(tokenizer.decode(gen_ids[0], skip_special_tokens=False))

            for cur_text, cur_label, gen_text in zip(
                texts, labels, gen_texts, strict=True
            ):
                examples.append(
                    {"text": cur_text, "label": cur_label, "generated": gen_text}
                )

    avg_loss = total_loss / len(dataloader) if len(dataloader) else 0
    return avg_loss, examples


def eval_and_log(
    model,
    dataloader: ScaledDataloader,
    tokenizer,
    introspection_prompt: str,
    device: str,
    sample_tables,
    samples_seen: int,
    max_generations: int,
    use_wandb: bool = True,
):
    val_loss, examples = evaluate(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        introspection_prompt=introspection_prompt,
        device=device,
        max_generations=max_generations,
    )
    print(f"Validation loss: {val_loss:.4f}")

    if use_wandb:
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

    if use_wandb and examples:
        for i, ex in enumerate(examples[:3]):
            if i in sample_tables and i < len(examples):
                sample_tables[i].add_data(
                    samples_seen, ex["text"], ex["label"], ex["generated"]
                )

    return val_loss, examples



def double_forward_epoch(
    model,
    tokenizer,
    optimizer,
    train_dataloader: ScaledDataloader,
    val_dataloader: ScaledDataloader,
    introspection_prompt: str,
    device: str,
    sample_tables,
    samples_seen: int,
    max_generations: int,
    use_wandb: bool = True,
):
    model.train()
    total_loss = 0
    batch_count = 0
    total_batches = len(train_dataloader)
    check_interval = max(1, total_batches // 2)

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        weight_diff_dict: dict[str, tuple[Tensor, Tensor]] = batch["weight_diff"]
        texts, labels = batch["text"], batch["label"]
        loraify_model_in_place(model, [weight_diff_dict])

        input_ids, labels_masked, attention_mask = build_prefix_inputs(
            tokenizer=tokenizer,
            labels=labels,
            introspection_prompt=introspection_prompt,
            device=device,
        )
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels_masked, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        samples_seen += len(texts)

        if use_wandb:
            wandb.log({"train_loss": loss, "total_samples": samples_seen})

        if (batch_idx + 1) % check_interval == 0:
            print(f"\nProgress: {batch_idx + 1}/{total_batches} batches")
            model.eval()
            eval_and_log(
                model=model,
                dataloader=val_dataloader,
                tokenizer=tokenizer,
                introspection_prompt=introspection_prompt,
                device=device,
                sample_tables=sample_tables,
                samples_seen=samples_seen,
                max_generations=max_generations,
                use_wandb=use_wandb,
            )
            model.train()

    return total_loss / total_batches, samples_seen


# %%
def main(
    model_name: str,
    input_dir: str,
    output_dir: str,
    bridges: Sequence[ResidAffineBridge],
    introspection_prompt: str = "What topic have you been trained on?",
    epochs: int = 5,
    batch_size: int = 8,
    max_generations: int = 16,
    lr: float = 1e-4,
    weight_diff_multiplier: float = 1.0,
    validation_split: float = 0.1,
    device: str | None = None,
    use_wandb: bool = False,
    wandb_name: str | None = None,
    debug: int = 0,
    train_size_div: int = 1,
):
    os.makedirs(output_dir, exist_ok=False)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if use_wandb:
        wandb.init(
            project="diff-interpretation",
            entity="atticusw",
            name=wandb_name,
            dir=output_dir,
            config={
                "model_name": model_name,
                "input_dir": input_dir,
                "bridges": [{
                    "read_layer": b.read_layer,
                    "write_layer": b.write_layer,
                    "rank": b.rank,
                    "init_A_std": b.init_A_std,
                } for b in bridges],
                "introspection_prompt": introspection_prompt,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    ).to(device)  # type: ignore

    for p in model.parameters():
        p.requires_grad = False

    trainable = []
    for bridge in bridges:
        for p in bridge.parameters():
            p.requires_grad = True
            trainable.append(p)

    print(f"Total number of trainable parameters: {sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    hooked_model = LoRADoubleForward(model, bridges)

    train_dataloader, val_dataloader = make_dataloaders(
        input_dir=input_dir,
        batch_size=batch_size,
        device=device,
        validation_split=validation_split,
        train_size_div=train_size_div,
        weight_diff_multiplier=weight_diff_multiplier,
        debug=debug,
    )
   
    samples_seen = 0
    sample_tables = {}

    val_loss, examples = evaluate(
        model=hooked_model,
        dataloader=val_dataloader,
        tokenizer=tokenizer,
        introspection_prompt=introspection_prompt,
        device=device,
        max_generations=max_generations,
    )
    print(f"Starting validation loss: {val_loss:.4f}")

    if use_wandb:
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

            train_loss, samples_seen = double_forward_epoch(
                model=hooked_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                introspection_prompt=introspection_prompt,
                device=device,
                sample_tables=sample_tables,
                samples_seen=samples_seen,
                max_generations=max_generations,
                use_wandb=use_wandb,
            )
            print(f"Train loss: {train_loss:.4f}")

            # save bridges checkpoint
            bridges_checkpoint_path = os.path.join(output_dir, f"bridges_ep_{epoch + 1}.pt")
            torch.save([bridge.state_dict() for bridge in bridges], bridges_checkpoint_path)
            print(f"Epoch {epoch + 1}/{epochs} complete; saved checkpoint.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    print("Generating final validation examples...")
    final_val_loss, final_examples = evaluate(
        model=hooked_model,
        dataloader=val_dataloader,
        introspection_prompt=introspection_prompt,
        device=device,
        tokenizer=tokenizer,
        max_generations=max_generations,
    )
    print(f"Final validation loss: {final_val_loss:.4f}")

    if use_wandb:
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




if __name__ == "__main__":
    from datetime import datetime

    def timestamp() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    model_name = "Qwen/Qwen3-4B"
    config = AutoConfig.from_pretrained(model_name)

    bridges = [
        ResidDiffBridge(d_model=config.hidden_size, rank=16, read_layer=L, write_layer=L)
        for L in range(config.num_hidden_layers)
    ]

    run_name = f"{timestamp()}-bridges-qwen3-4b"

    main(
        model_name=model_name,
        input_dir="/workspace/diff-interpretation-tuning/data/loras/hidden-topic/qwen3-4b/weight-diffs",
        output_dir=f"/workspace/diff-interpretation-tuning/results/{run_name}",
        bridges = bridges,
        introspection_prompt="What topic have you been trained on?",
        epochs=4,
        batch_size=8,
        max_generations=8,
        wandb_name=run_name,
        use_wandb=True,
        debug=0,
    )

    # model_name = "Qwen/Qwen3-4B"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to("cuda:0")
    # base.eval()

    # # minimal dummy bridges that should be zero-effect
    # config = base.config
    # bridges = [
    #     ResidDiffBridge(d_model=config.hidden_size, rank=8, read_layer=L, write_layer=L)
    #     for L in range(25, 32)
    # ]
    # wrapped = LoRADoubleForward(base, bridges).cuda()
    # wrapped.eval()

    # inputs = tokenizer.apply_chat_template(
    #     [{"role": "user", "content": "What topic have you been trained on?"}],
    #     add_generation_prompt=True,
    #     enable_thinking=False,
    #     tokenize=True,
    #     return_tensors="pt",
    # ).to(base.device)

    # with torch.no_grad():
    #     out_base = base(inputs).logits
    #     out_wrap = wrapped(inputs).logits

    # print("max abs diff:", (out_wrap - out_base).abs().max().item())