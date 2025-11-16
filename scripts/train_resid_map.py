import torch
from torch import Tensor
from torch import nn
from train_weight_to_text import load_training_data, build_prefix_inputs



class ResidualAffine(nn.Module):
    def __init__(self, d_model: int, rank: int|None, init_A_std: float=1e-3):
        super().__init__()
        if rank is None:
            rank = d_model
        self.proj_A = nn.Linear(d_model, rank, bias=False)
        self.proj_B = nn.Linear(rank, d_model, bias=True)

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
        return acts + self.proj_B(self.proj_A(self._read))


# TODO: better type hints: adapter should have read() and write() methods
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


def train_epoch(model):
    ...