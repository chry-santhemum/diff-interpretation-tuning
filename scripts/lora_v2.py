import math
import collections
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from turtle import setworldcoordinates
from typing import Any

import torch
import torch.nn as nn
from transformers import Gemma3ForConditionalGeneration
from transformers import GenerationMixin, GenerationConfig

class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        As: list[torch.Tensor],
        Bs: list[torch.Tensor],
    ):
        super().__init__()
        self.base_layer = base_layer

        self.lora_enabled = True

        # Task-specific LoRA parameters
        self.As = [
            A.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
            for A in As
        ]
        self.Bs = [
            B.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
            for B in Bs
        ]

    def forward(self, x):
        base_output = self.base_layer(x)
        if not self.lora_enabled:
            return base_output

        lora_output = 0
        for i in range(len(self.As)):
            A = self.As[i]
            B = self.Bs[i]

            rank = B.shape[-1]

            middle = torch.einsum("b...i,bir->b...r", x, B)
            lora_output += torch.einsum("b...r,bro->b...o", middle, A) / rank

        return base_output + lora_output


def disable_lora_in_place(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_enabled = False


def enable_lora_in_place(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_enabled = True


def _set_module(model: nn.Module, layer_name: str, new_mod: nn.Module):
    parts = layer_name.split(".")

    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)

    setattr(parent, parts[-1], new_mod)


def loraify_model_in_place(
    model: nn.Module,
    lora_param_dicts: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
) -> nn.Module:
    """
    Replace all nn.Linear layers in the model with Lora layers and apply the given LoRA(s).
    Modifies the model in place.

    Can be called multiple times. Every call will remove previously applied LoRAs.
    """
    # First unwrap any LoRALinear layers
    for layer_name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            original = module.base_layer
            _set_module(model, layer_name, original)

    # Collect all LoRA parameters for each layer
    layer_to_As_and_Bs = collections.defaultdict(list)
    for lora_param_dict in lora_param_dicts:
        for name, (A, B) in lora_param_dict.items():
            layer_to_As_and_Bs[name].append((A.detach().clone(), B.detach().clone()))

    # Now wrap all Linear layers with new LoRALinear
    layer_name_to_module = dict(model.named_modules())

    for layer_name, As_and_Bs in layer_to_As_and_Bs.items():
        if isinstance(model, Gemma3ForConditionalGeneration):
            if layer_name.startswith("language_model.model."):
                # A hack to fix updated model format for gemma3-4-b-it
                layer_name = layer_name.replace(
                    "language_model.model.", "model.language_model."
                )

        module = layer_name_to_module["model." + layer_name]  # TODO: fix this hack
        assert isinstance(module, nn.Linear)
        wrapped = LoRALinear(
            module, [A for A, _ in As_and_Bs], [B for _, B in As_and_Bs]
        )
        _set_module(model, "model." + layer_name, wrapped)

    torch.cuda.empty_cache()
    return model



class ResidAffineBridge(nn.Module):
    def __init__(
        self, 
        d_model: int,
        rank: int | None,
        read_layer: int,
        write_layer: int,
        init_A_std: float = 1e-3,
        alpha: float = 1.0,
        device: str | None = None,
    ):
        super().__init__()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        if rank is None:
            rank = d_model
        self.rank = rank
        self.d_model = d_model
        self.proj_A = nn.Linear(d_model, rank, bias=False).to(device=device)
        self.proj_B = nn.Linear(rank, d_model, bias=True).to(device=device)
        self.scaling_factor = alpha / math.sqrt(rank)
        self.init_A_std = init_A_std

        # Initialization:
        # proj_A ~ N(0, init_A_std), proj_B = 0
        with torch.no_grad():
            nn.init.normal_(self.proj_A.weight, mean=0.0, std=init_A_std)
            nn.init.zeros_(self.proj_B.weight)
            nn.init.zeros_(self.proj_B.bias)

        self.read_layer = read_layer
        self.write_layer = write_layer

    def forward(  # type: ignore[override]
        self,
        lora_enabled_act: torch.Tensor,
        lora_disabled_act: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the residual tensor (to be added to the lora-disabled activations).
        """
        raise NotImplementedError


class ResidDiffBridge(ResidAffineBridge):
    def forward(  # type: ignore[override]
        self,
        lora_enabled_act: torch.Tensor,
        lora_disabled_act: torch.Tensor,
    ) -> torch.Tensor:
        delta = lora_enabled_act - lora_disabled_act
        # return self.scaling_factor * self.proj_B(self.proj_A(delta)) + delta
        return self.scaling_factor * self.proj_B(self.proj_A(delta))


class ResidDirectBridge(ResidAffineBridge):
    def forward(  # type: ignore[override]
        self,
        lora_enabled_act: torch.Tensor,
        lora_disabled_act: torch.Tensor,
    ) -> torch.Tensor:
        return self.scaling_factor * self.proj_B(self.proj_A(lora_enabled_act))


class LoRADoubleForward(nn.Module, GenerationMixin):
    """
    Runs a LoRA-enabled pass to capture activations, 
    then a LoRA-disabled pass while injecting learned bridge modules.
    """
    # Hugging Face generation utilities expect this flag on model classes.
    _is_stateful: bool = False

    def __init__(
        self,
        model: nn.Module,
        bridges: Sequence[ResidAffineBridge],
    ):
        super().__init__()
        self.model = model
        disable_lora_in_place(self.model)
        self.bridges = bridges

        # ensure device and dtype of bridges agree with the model
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        for b in bridges:
            b.to(device=device, dtype=dtype)

        self.lora_enabled_acts: dict[int, torch.Tensor] = {}
        self.lora_disabled_acts: dict[int, torch.Tensor] = {}

        # Make HF generation happy
        self.config = model.config
        self.main_input_name = getattr(model, "main_input_name", "input_ids")
        if hasattr(model, "generation_config") and model.generation_config is not None:
            self.generation_config = model.generation_config
        else:
            self.generation_config = GenerationConfig.from_model_config(model.config)  # type: ignore[arg-type]

    @property
    def device(self): return self.model.device

    @property
    def dtype(self): return self.model.dtype

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs) # type: ignore

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.model._reorder_cache(past_key_values, beam_idx) # type: ignore

    def get_output_embeddings(self):
        return self.model.get_output_embeddings() # type: ignore

    def can_generate(self):
        return True


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        *args,
        **kwargs,
    ):
        # Ensure input_ids / attention_mask get forwarded, while also satisfying
        # HF GenerationMixin's signature inspection for generate().
        if input_ids is not None:
            kwargs.setdefault("input_ids", input_ids)
        if attention_mask is not None:
            kwargs.setdefault("attention_mask", attention_mask)

        enable_lora_in_place(self.model)
        capture_handles = []
        for b in self.bridges:
            if b.read_layer >= self.model.config.num_hidden_layers:  # type: ignore
                raise ValueError(f"Read layer {b.read_layer} not found in model.")
            if b.write_layer >= self.model.config.num_hidden_layers:  # type: ignore
                raise ValueError(f"Write layer {b.write_layer} not found in model.")

            module: nn.Module = self.model.model.layers[b.read_layer]  # type: ignore
            capture_handles.append(module.register_forward_hook(self._capture_hook(b, self.lora_enabled_acts)))

        self.model(*args, **kwargs)
        self._remove_handles(capture_handles)

        disable_lora_in_place(self.model)
        capture_handles = []
        for b in self.bridges:
            module: nn.Module = self.model.model.layers[b.read_layer]  # type: ignore
            capture_handles.append(module.register_forward_hook(self._capture_hook(b, self.lora_disabled_acts)))

        self.model(*args, **kwargs)
        self._remove_handles(capture_handles)

        inject_handles = []
        for b in self.bridges:
            module: nn.Module = self.model.model.layers[b.write_layer]  # type: ignore
            inject_handles.append(module.register_forward_hook(self._inject_hook(b)))

        lora_interpreter_output = self.model(*args, **kwargs)
        self._remove_handles(inject_handles)

        # Cleanup
        enable_lora_in_place(self.model)
        self.lora_enabled_acts.clear()
        self.lora_disabled_acts.clear()

        return lora_interpreter_output


    def _capture_hook(self, bridge: ResidAffineBridge, record_dict: dict):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            tensor = hidden_states.detach()
            record_dict[bridge.read_layer] = tensor
            return output
        return hook

    def _inject_hook(self, bridge: ResidAffineBridge):
        """
        Just adds activation.
        """
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                lora_interpreted = output[0]
            else:
                lora_interpreted = output

            adapter_output = bridge(self.lora_enabled_acts[bridge.read_layer], self.lora_disabled_acts[bridge.read_layer])
            updated = lora_interpreted + adapter_output
            
            if isinstance(output, tuple):
                return (updated, *output[1:])
            else:
                return updated
        return hook

    @staticmethod
    def _remove_handles(handles):
        for handle in handles:
            handle.remove()
