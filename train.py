import argparse
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from model.transformer_lm import TransformerLM, ModelConfig


def get_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_tokens(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Token file not found: {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size: int, block_size: int, device: str):
    """
    Randomly sample token chunks from a continuous token stream.

    x: [B, T]
    y: [B, T], shifted by one token
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )

    x = x.to(device)
    y = y.to(device)
    return x, y


def get_lr(step: int, max_steps: int, learning_rate: float, warmup_steps: int):
    """
    Warmup + cosine decay learning rate schedule.
    """
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * learning_rate * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    model.eval()

    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(cfg["training"]["eval_iters"]):
            x, y = get_batch(
                data=data,
                batch_size=cfg["training"]["batch_size"],
                block_size=cfg["model"]["block_size"],
                device=device,
            )
            _, loss = model(x, y)
            losses.append(loss.item())

        out[split] = float(np.mean(losses))

    model.train()
    return out


def save_checkpoint(model, optimizer, step, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": cfg,
    }
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tinystories_mps_debug.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["system"].get("seed", 42)
    set_seed(seed)

    device = get_device(cfg["system"].get("device", "auto"))
    print(f"Using device: {device}")

    train_data = load_tokens(cfg["data"]["train_bin"])
    val_data = load_tokens(cfg["data"]["val_bin"])

    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")

    model_cfg = ModelConfig(**cfg["model"])
    model = TransformerLM(model_cfg).to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        betas=(cfg["training"]["beta1"], cfg["training"]["beta2"]),
        weight_decay=cfg["training"]["weight_decay"],
    )

    log_dir = cfg["training"].get("log_dir", "runs/tinystories")
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_dir = cfg["training"].get("checkpoint_dir", "checkpoints/tinystories")
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = cfg["training"]["batch_size"]
    block_size = cfg["model"]["block_size"]
    grad_accum = cfg["training"]["gradient_accumulation_steps"]
    max_steps = cfg["training"]["max_steps"]
    grad_clip = cfg["training"]["grad_clip"]

    print("Start training...")
    model.train()

    last_time = time.time()

    for step in range(max_steps + 1):
        lr = get_lr(
            step=step,
            max_steps=max_steps,
            learning_rate=cfg["training"]["learning_rate"],
            warmup_steps=cfg["training"]["warmup_steps"],
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0

        for micro_step in range(grad_accum):
            x, y = get_batch(
                data=train_data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
            )

            _, loss = model(x, y)
            loss = loss / grad_accum
            loss.backward()
            total_loss += loss.item()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        now = time.time()
        dt = now - last_time
        last_time = now

        tokens_per_step = batch_size * block_size * grad_accum
        tokens_per_second = tokens_per_step / max(dt, 1e-8)

        if step % cfg["training"]["log_interval"] == 0:
            print(
                f"step {step:5d} | "
                f"loss {total_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"dt {dt:.2f}s | "
                f"tokens/s {tokens_per_second:.0f}"
            )

            writer.add_scalar("train/loss", total_loss, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/tokens_per_second", tokens_per_second, step)

        if step % cfg["training"]["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)

            train_loss = losses["train"]
            val_loss = losses["val"]
            val_ppl = math.exp(min(val_loss, 20))

            print(
                f"[eval] step {step:5d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"val ppl {val_ppl:.2f}"
            )

            writer.add_scalar("eval/train_loss", train_loss, step)
            writer.add_scalar("eval/val_loss", val_loss, step)
            writer.add_scalar("eval/val_perplexity", val_ppl, step)

            ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
            save_checkpoint(model, optimizer, step, cfg, ckpt_path)

    final_path = os.path.join(checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, max_steps, cfg, final_path)

    writer.close()
    print(f"Training finished. Final checkpoint saved to: {final_path}")


if __name__ == "__main__":
    main()
