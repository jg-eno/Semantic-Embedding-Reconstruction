"""
Usage:
    python main.py
    python main.py --sentence "Your sentence here" --n 5
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

from config import MAX_NEW_TOKENS, SAMPLE_PARAGRAPH
from sampler import Sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate n diverse reconstructions from a paragraph embedding."
    )
    parser.add_argument("--sentence", type=str, default=SAMPLE_PARAGRAPH)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    kwargs: dict = {"max_new_tokens": args.max_new_tokens}
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature

    sampler = Sampler(**kwargs)

    print(f"\nInput    : {args.sentence!r}")
    print(f"Samples  : {args.n}\n")

    result = sampler.sample(args.sentence, n=args.n)

    print("\n── Results ──────────────────────────────────────────────────")
    for i, text in enumerate(result.samples, 1):
        print(f"  [{i}] {text}")
    print("─────────────────────────────────────────────────────────────\n")

    sampler.unload()


if __name__ == "__main__":
    main()
