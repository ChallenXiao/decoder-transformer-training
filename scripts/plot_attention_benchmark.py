import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/attention_benchmark_mps.csv")
    parser.add_argument("--out_dir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Plot 1: step time
    plt.figure()
    for impl in df["attn_impl"].unique():
        sub = df[df["attn_impl"] == impl]
        plt.plot(sub["seq_len"], sub["avg_step_time_ms"], marker="o", label=impl)

    plt.xlabel("Sequence Length")
    plt.ylabel("Average Step Time (ms)")
    plt.title("Attention Benchmark: Step Time vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "attention_step_time.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: tokens/s
    plt.figure()
    for impl in df["attn_impl"].unique():
        sub = df[df["attn_impl"] == impl]
        plt.plot(sub["seq_len"], sub["tokens_per_second"], marker="o", label=impl)

    plt.xlabel("Sequence Length")
    plt.ylabel("Tokens / Second")
    plt.title("Attention Benchmark: Tokens/s vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "attention_tokens_per_second.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 3: memory
    plt.figure()
    for impl in df["attn_impl"].unique():
        sub = df[df["attn_impl"] == impl]
        plt.plot(sub["seq_len"], sub["memory_mb"], marker="o", label=impl)

    plt.xlabel("Sequence Length")
    plt.ylabel("Allocated Memory (MB)")
    plt.title("Attention Benchmark: Memory vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "attention_memory.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
