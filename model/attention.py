import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import build_rope_cache, apply_rope


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional naive or SDPA backend.
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

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        if attn_impl not in {"naive", "sdpa"}:
            raise ValueError("attn_impl must be either 'naive' or 'sdpa'.")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.dropout = dropout
        self.attn_impl = attn_impl

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.block_size}."
            )

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # [B, T, C] -> [B, H, T, D]
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K.
        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.attn_impl == "naive":
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = self.causal_mask[:, :, :seq_len, :seq_len]
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        else:
            # PyTorch scaled dot product attention.
            # dropout_p should be 0 during evaluation.
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        # [B, H, T, D] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y