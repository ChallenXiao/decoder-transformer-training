import argparse
import torch
from transformers import AutoTokenizer

from model.transformer_lm import TransformerLM, ModelConfig


def get_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tinystories_mps_debug/final.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    cfg = checkpoint["config"]
    model_cfg = ModelConfig(**cfg["model"])

    model = TransformerLM(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    text = tokenizer.decode(output_ids[0].tolist())

    print("\n========== Prompt ==========")
    print(args.prompt)

    print("\n========== Generated Text ==========")
    print(text)


if __name__ == "__main__":
    main()