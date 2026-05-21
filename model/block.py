import torch
import torch.nn as nn

from model.rmsnorm import RMSNorm
from model.attention import CausalSelfAttention
from model.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    LLaMA-style pre-norm Transformer decoder block.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        attn_impl: str = "naive",
    ):
        super().__init__()

        self.attn_norm = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
            attn_impl=attn_impl,
        )

        self.ffn_norm = RMSNorm(n_embd)
        self.ffn = SwiGLU(
            dim=n_embd,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x