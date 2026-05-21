import argparse
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    batch_size: int
    dtype_bytes: int = 4
    optimizer_state_bytes: int = 4


def estimate_num_params(cfg: MemoryConfig) -> int:
    """
    Rough parameter estimation for a LLaMA-style decoder-only Transformer.

    Components:
    - token embedding
    - attention qkv projection
    - attention output projection
    - SwiGLU FFN: gate, up, down
    - RMSNorm weights
    - final RMSNorm
    lm_head is tied with token embedding, so not counted separately.
    """

    vocab_embed = cfg.vocab_size * cfg.n_embd

    # Attention per layer:
    # qkv: n_embd -> 3 * n_embd
    # out: n_embd -> n_embd
    attn_per_layer = cfg.n_embd * 3 * cfg.n_embd + cfg.n_embd * cfg.n_embd

    # SwiGLU hidden dim, same as implementation
    hidden_dim = int(8 * cfg.n_embd / 3)
    multiple_of = 256
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    # gate_proj, up_proj, down_proj
    ffn_per_layer = (
        cfg.n_embd * hidden_dim
        + cfg.n_embd * hidden_dim
        + hidden_dim * cfg.n_embd
    )

    # RMSNorm: attn_norm + ffn_norm per layer, final_norm once
    norm_params = cfg.n_layer * 2 * cfg.n_embd + cfg.n_embd

    total = vocab_embed + cfg.n_layer * (attn_per_layer + ffn_per_layer) + norm_params
    return total


def bytes_to_mb(x: float) -> float:
    return x / 1024**2


def bytes_to_gb(x: float) -> float:
    return x / 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dtype_bytes", type=int, default=4)
    args = parser.parse_args()

    cfg = MemoryConfig(**vars(args))

    n_params = estimate_num_params(cfg)

    param_mem = n_params * cfg.dtype_bytes
    grad_mem = n_params * cfg.dtype_bytes

    # AdamW keeps first moment and second moment, commonly fp32.
    adam_mem = n_params * 2 * cfg.optimizer_state_bytes

    # Very rough activation estimation:
    # hidden states per layer: B * T * C
    activation_mem = (
        cfg.batch_size
        * cfg.block_size
        * cfg.n_embd
        * cfg.n_layer
        * cfg.dtype_bytes
    )

    # Attention score matrix:
    # B * H * T * T
    attention_score_mem = (
        cfg.batch_size
        * cfg.n_head
        * cfg.block_size
        * cfg.block_size
        * cfg.dtype_bytes
    )

    total_estimated = (
        param_mem
        + grad_mem
        + adam_mem
        + activation_mem
        + attention_score_mem
    )

    print("\n========== Model Configuration ==========")
    print(f"vocab_size: {cfg.vocab_size}")
    print(f"block_size: {cfg.block_size}")
    print(f"n_layer:    {cfg.n_layer}")
    print(f"n_head:     {cfg.n_head}")
    print(f"n_embd:     {cfg.n_embd}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"dtype:      {cfg.dtype_bytes} bytes")

    print("\n========== Parameter Estimation ==========")
    print(f"Estimated parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    print("\n========== Training Memory Estimation ==========")
    print(f"Parameters:        {bytes_to_mb(param_mem):8.2f} MB")
    print(f"Gradients:         {bytes_to_mb(grad_mem):8.2f} MB")
    print(f"AdamW states:      {bytes_to_mb(adam_mem):8.2f} MB")
    print(f"Activations rough: {bytes_to_mb(activation_mem):8.2f} MB")
    print(f"Attention scores:  {bytes_to_mb(attention_score_mem):8.2f} MB")

    print("------------------------------------------")
    print(f"Total rough estimate: {bytes_to_mb(total_estimated):8.2f} MB")
    print(f"Total rough estimate: {bytes_to_gb(total_estimated):8.3f} GB")

    print("\n========== Key Insight ==========")
    print(
        "Training memory is much larger than parameter memory because gradients, "
        "AdamW optimizer states, activations, and attention score tensors must also be stored."
    )
    print(
        "The attention score tensor scales as O(batch_size * n_head * seq_len^2), "
        "which becomes a major bottleneck for long-context training."
    )


if __name__ == "__main__":
    main()
