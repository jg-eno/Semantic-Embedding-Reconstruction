"""
Token-length EDA for abisee/cnn_dailymail (train/validation/test).

Computes mean/median/mode/percentiles of article token length per split,
to inform the packing/truncation cutoff for the reconstruction dataset.
"""

import statistics

import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "abisee/cnn_dailymail"
SUBSET = "3.0.0"
SPLITS = ["train", "validation", "test"]
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 256          # tokenizer batch size, not related to training batch size
NUM_PROC = 4              # parallel workers for tokenization; set to your core count

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def token_lengths_for_split(split):
    """Return a list of token counts, one per article, for a given split."""
    dataset = load_dataset(DATASET_NAME, split=split, name=SUBSET)  # full, not streaming — need every length

    def add_length(batch):
        ids = tokenizer(batch["article"], add_special_tokens=False)["input_ids"]
        return {"num_tokens": [len(x) for x in ids]}

    dataset = dataset.map(
        add_length,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=dataset.column_names,
    )
    return dataset["num_tokens"]


def summarize(lengths, split_name):
    mode_val = statistics.mode(lengths)  # single most common exact token count — see note below
    quantiles = statistics.quantiles(lengths, n=100)  # p1..p99

    print(f"\n--- {split_name} (n={len(lengths)}) ---")
    print(f"mean:   {statistics.mean(lengths):.1f}")
    print(f"median: {statistics.median(lengths)}")
    print(f"mode:   {mode_val}  (exact-value mode; usually not very informative for continuous-ish data)")
    print(f"min:    {min(lengths)}")
    print(f"max:    {max(lengths)}")
    print(f"p50:    {quantiles[49]}")
    print(f"p90:    {quantiles[89]}")
    print(f"p95:    {quantiles[94]}")
    print(f"p99:    {quantiles[98]}")

    pct_over_512 = sum(1 for x in lengths if x > 512) / len(lengths) * 100
    print(f"% of articles > 512 tokens: {pct_over_512:.1f}%")


def plot_histogram(all_lengths_by_split, out_path="token_length_histogram.png"):
    plt.figure(figsize=(10, 6))
    for split_name, lengths in all_lengths_by_split.items():
        plt.hist(lengths, bins=100, alpha=0.5, label=split_name)
    plt.axvline(512, color="red", linestyle="--", label="512 cutoff")
    plt.xlabel("Token length")
    plt.ylabel("Article count")
    plt.title("Article token length distribution — abisee/cnn_dailymail")
    plt.legend()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nHistogram saved to {out_path}")


if __name__ == "__main__":
    lengths_by_split = {}
    for split in SPLITS:
        lengths = token_lengths_for_split(split)
        summarize(lengths, split)
        lengths_by_split[split] = lengths

    plot_histogram(lengths_by_split)