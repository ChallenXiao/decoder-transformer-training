import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    LLaMA-style feed-forward network using SwiGLU activation.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            # LLaMA-style FFN dimension: about 8/3 * dim, rounded up.
            hidden_dim = int(8 * dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)