import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--train_samples", type=int, default=50000)
    parser.add_argument("--val_samples", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="data/processed/tinystories")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    eos_id = tokenizer.eos_token_id
    print(f"EOS token id: {eos_id}")

    train_texts = dataset["train"]["text"][: args.train_samples]
    val_texts = dataset["validation"]["text"][: args.val_samples]

    def tokenize_texts(texts, split_name):
        all_ids = []

        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)
            ids.append(eos_id)
            all_ids.extend(ids)

            if (i + 1) % 5000 == 0:
                print(f"{split_name}: tokenized {i + 1} samples")

        arr = np.array(all_ids, dtype=np.uint16)
        print(f"{split_name}: total tokens = {len(arr):,}")
        return arr

    train_ids = tokenize_texts(train_texts, "train")
    val_ids = tokenize_texts(val_texts, "val")

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Saved train tokens to: {train_path}")
    print(f"Saved val tokens to: {val_path}")
    print("Data preparation finished.")


if __name__ == "__main__":
    main()