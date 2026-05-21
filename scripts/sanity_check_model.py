import torch

from model.transformer_lm import TransformerLM, ModelConfig


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()
    print(f"Using device: {device}")

    config = ModelConfig(
        vocab_size=50257,
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
        attn_impl="naive",
    )

    model = TransformerLM(config).to(device)
    model.train()

    batch_size = 2
    seq_len = 64

    idx = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )

    logits, loss = model(idx, targets)
    print("logits shape:", logits.shape)
    print("loss:", loss.item())

    loss.backward()
    print("backward: ok")

    model.eval()
    prompt = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, 8),
        device=device,
    )
    out = model.generate(prompt, max_new_tokens=8, top_k=50)
    print("generated token shape:", out.shape)
    print("sanity check passed.")


if __name__ == "__main__":
    main()
