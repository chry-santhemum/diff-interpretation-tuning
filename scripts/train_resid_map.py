# %%
import math
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import wandb

from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_recovery.multi_lora import ScaledDataloader, multi_loraify_model
from train_weight_to_text import evaluate, load_training_data, train_epoch
from lora_v2 import loraify_model_in_place

class ResidualAffine(nn.Module):
    def __init__(
        self,
        d_model: int,
        rank: int | None,
        init_A_std: float = 1e-3,
        alpha: float = 1.0,
    ):
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
    read_block = model.model.layers[read_layer]
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
        write_block = model.model.layers[write_layer]
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


def main(
    model_name: str,
    input_dir: str,
    output_dir: str,
    read_layer: int,
    write_layer: int,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-4,
    rank: int = 32,
    init_A_std: float = 1e-3,
    alpha: float = 1.0,
    weight_diff_multiplier: float = 1.0,
    validation_split: float = 0.1,
    device: str | None = None,
    use_wandb: bool = False,
    wandb_name: str = None,
    introspection_prompt: str = "What topic have you been trained on?",
    debug: bool = False,
    train_size_div: int = 1,
):
    os.makedirs(output_dir, exist_ok=False)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup wandb: TODO
    if use_wandb:
        wandb.init(
            project="diff-interpretation",
            entity="atticusw",
            name=wandb_name,
            # config=vars(locals()),
            dir=output_dir,
        )

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
        rank=rank,
        init_A_std=init_A_std,
        alpha=alpha,
    ).to(device=device, dtype=model_dtype)

    for p in adapter.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)

    apply_resid_map(model, adapter, read_layer=read_layer, write_layer=write_layer)

    # Load Data
    all_data = load_training_data(input_dir=input_dir, debug=debug)
    random.seed(42)
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

    samples_seen = 0
    sample_tables = {}
    val_loss, examples = evaluate(
        model=model,
        dataloader=val_dataloader,
        introspection_prompt=introspection_prompt,
        device=device,
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        prefix_token_len=prefix_token_len,
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

            train_loss, samples_seen = train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                val_dataloader=val_dataloader,
                sample_tables=sample_tables,
                introspection_prompt=introspection_prompt,
                device=device,
                tokenizer=tokenizer,
                prefix_tokens=prefix_tokens,
                prefix_token_len=prefix_token_len,
                samples_seen=samples_seen,
                use_wandb=use_wandb,
            )
            print(f"Train loss: {train_loss:.4f}")

            # save adapter checkpoint
            adapter_checkpoint_path = os.path.join(output_dir, f"adapter_ep_{epoch + 1}.pt")
            torch.save(adapter.state_dict(), adapter_checkpoint_path)
            print(f"Epoch {epoch + 1}/{epochs} complete; saved checkpoint.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    print("Generating final validation examples...")
    final_val_loss, final_examples = evaluate(
        model=model,
        dataloader=val_dataloader,
        introspection_prompt=introspection_prompt,
        device=device,
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        prefix_token_len=prefix_token_len,
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
    import argparse

    def timestamp() -> str:
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--read_layer", type=int, required=True)
    parser.add_argument("--write_layer", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    
    args = parser.parse_args()

    read_layer = args.read_layer
    write_layer = args.write_layer
    rank = args.rank

    run_name = f"{timestamp()}-backdoor-r{read_layer}w{write_layer}-affine-rk{rank}-qwen3-4b"

    main(
        model_name="Qwen/Qwen3-4B",
        input_dir="/workspace/diff-interpretation-tuning/data/loras/hidden-topic/qwen3-4b/weight-diffs",
        output_dir=f"/workspace/diff-interpretation-tuning/results/{run_name}",
        read_layer=read_layer,
        write_layer=write_layer,
        rank=rank,
        epochs=4,
        wandb_name=run_name,
        use_wandb=True,
        debug=False,
    )

