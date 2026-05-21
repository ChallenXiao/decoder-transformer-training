import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
):
    """
    Build RoPE cos/sin cache.

    Returns:
        cos, sin: [1, 1, seq_len, head_dim // 2]
    """
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE.")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)

    cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
    sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding.

    Args:
        x: [batch, n_head, seq_len, head_dim]
        cos/sin: [1, 1, seq_len, head_dim // 2]
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    x_out = torch.stack((x_rot_even, x_rot_odd), dim=-1)
    return x_out.flatten(-2)