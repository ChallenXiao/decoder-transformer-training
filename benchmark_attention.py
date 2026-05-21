import argparse
import csv
import os
import time
from dataclasses import asdict

import torch

from model.attention import CausalSelfAttention


def get_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def reset_memory_stats(device: str):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def get_memory_mb(device: str) -> float:
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2

    if device == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024**2
        except Exception:
            return -1.0

    return -1.0


def benchmark_one(
    device: str,
    attn_impl: str,
    batch_size: int,
    seq_len: int,
    n_embd: int,
    n_head: int,
    dropout: float,
    warmup: int,
    iters: int,
):
    torch.manual_seed(42)

    attn = CausalSelfAttention(
        n_embd=n_embd,
        n_head=n_head,
        block_size=seq_len,
        dropout=dropout,
        bias=False,
        attn_impl=attn_impl,
    ).to(device)

    attn.train()

    x = torch.randn(batch_size, seq_len, n_embd, device=device)

    # Warmup
    for _ in range(warmup):
        x_in = x.detach().clone().requires_grad_(True)
        y = attn(x_in)
        loss = y.pow(2).mean()
        loss.backward()
        synchronize(device)

    reset_memory_stats(device)
    synchronize(device)

    start = time.perf_counter()

    for _ in range(iters):
        x_in = x.detach().clone().requires_grad_(True)
        y = attn(x_in)
        loss = y.pow(2).mean()
        loss.backward()
        synchronize(device)

    end = time.perf_counter()

    total_time = end - start
    avg_step_time = total_time / iters
    tokens_per_step = batch_size * seq_len
    tokens_per_second = tokens_per_step / avg_step_time
    memory_mb = get_memory_mb(device)

    return {
        "device": device,
        "attn_impl": attn_impl,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_embd": n_embd,
        "n_head": n_head,
        "head_dim": n_embd // n_head,
        "avg_step_time_ms": avg_step_time * 1000,
        "tokens_per_second": tokens_per_second,
        "memory_mb": memory_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seq_lens", type=str, default="64,128,256")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=str, default="results/attention_benchmark.csv")
    args = parser.parse_args()

    device = get_device(args.device)
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print(f"Using device: {device}")
    print(f"seq_lens: {seq_lens}")
    print(f"batch_size: {args.batch_size}, n_embd: {args.n_embd}, n_head: {args.n_head}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results = []

    for seq_len in seq_lens:
        for attn_impl in ["naive", "sdpa"]:
            print(f"\nBenchmarking impl={attn_impl}, seq_len={seq_len}")

            try:
                result = benchmark_one(
                    device=device,
                    attn_impl=attn_impl,
                    batch_size=args.batch_size,
                    seq_len=seq_len,
                    n_embd=args.n_embd,
                    n_head=args.n_head,
                    dropout=args.dropout,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                results.append(result)

                print(
                    f"impl={attn_impl:5s} | "
                    f"seq={seq_len:4d} | "
                    f"step={result['avg_step_time_ms']:.2f} ms | "
                    f"tokens/s={result['tokens_per_second']:.0f} | "
                    f"memory={result['memory_mb']:.2f} MB"
                )

            except Exception as e:
                print(f"Failed impl={attn_impl}, seq_len={seq_len}: {e}")

    if results:
        fieldnames = list(results[0].keys())

        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nSaved benchmark results to: {args.out}")

    print("\nDone.")


if __name__ == "__main__":
    main()