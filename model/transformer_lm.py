from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rmsnorm import RMSNorm
from model.block import TransformerBlock


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False
    attn_impl: str = "naive"


class TransformerLM(nn.Module):
    """
    Minimal LLaMA-style decoder-only language model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    block_size=config.block_size,
                    dropout=config.dropout,
                    bias=config.bias,
                    attn_impl=config.attn_impl,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying, commonly used in language models.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        Args:
            idx: [batch, seq_len]
            targets: [batch, seq_len]
        """
        batch_size, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Input seq_len {seq_len} exceeds block_size {self.config.block_size}."
            )

        x = self.token_embedding(idx)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx