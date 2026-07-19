"""
Sampler: encodes a sentence, runs the inverter n times with escalating
temperature, deduplicates near-matches, and returns the most diverse results.
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import torch

from config import CONFIG, DEVICE, MAX_NEW_TOKENS, SAMPLER_DEFAULTS
from factory import InverterFactory
from models.encoder import Encoder

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class SampleResult:
    sentence: str
    embedding: torch.Tensor   # (1, D) on CPU
    samples: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [f"SampleResult for: {self.sentence!r}", "  samples:"]
        for i, s in enumerate(self.samples, 1):
            lines.append(f"    [{i}] {s!r}")
        return "\n".join(lines)


def _flush_vram() -> None:
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


class Sampler:
    """
    End-to-end sampling pipeline.

    Memory layout on a small GPU
    ─────────────────────────────
    1. Encoder loads on GPU → encode() → embedding moved to CPU immediately
    2. Encoder model fully destroyed (del + gc + empty_cache) — not just offloaded
    3. Inverter loads on GPU and stays resident for all n generate() calls

    Diversity strategy: sample i uses temperature + i * temperature_step,
    so early passes are conservative and later passes are more exploratory.
    Near-duplicates are removed via Jaccard similarity, then results are
    ranked by greedy max-dissimilarity.
    """

    def __init__(
        self,
        cfg: dict | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = SAMPLER_DEFAULTS["temperature"],
        top_p: float = SAMPLER_DEFAULTS["top_p"],
        top_k: int = SAMPLER_DEFAULTS["top_k"],
        repetition_penalty: float = SAMPLER_DEFAULTS["repetition_penalty"],
        temperature_step: float = SAMPLER_DEFAULTS["temperature_step"],
        dedup_threshold: float = 0.85,
    ):
        self.cfg = cfg or CONFIG
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.temperature_step = temperature_step
        self.dedup_threshold = dedup_threshold

        self._encoder: Encoder | None = None
        self._inverter = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        if self._encoder is None:
            print("[Sampler] Loading encoder …")
            self._encoder = Encoder()

    def _destroy_encoder(self) -> None:
        """Fully free the encoder's GPU memory before loading the inverter."""
        if self._encoder is not None and self._tokenizer is None:
            self._tokenizer = self._encoder.tokenizer
        if self._encoder is not None and self._encoder.model is not None:
            del self._encoder.model
            self._encoder.model = None
            _flush_vram()

    def _load_inverter(self, paragraph_dim: int) -> None:
        if self._inverter is None:
            self._destroy_encoder()
            print("[Sampler] Loading inverter …")
            self._inverter = InverterFactory().load(
                repo=self.cfg["repo"],
                filename=self.cfg["filename"],
                paragraph_dim=paragraph_dim,
            )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, sentence: str) -> torch.Tensor:
        """Encode and immediately move the result to CPU to free GPU activations."""
        embedding = self._encoder.encode([sentence]).cpu()
        _flush_vram()
        return embedding

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _diverse_generate(self, embedding: torch.Tensor, n: int) -> list[str]:
        results = []
        for i in range(n):
            generated = self._inverter.generate(
                paragraph_embs=embedding,
                tokenizer=self._tokenizer,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature + i * self.temperature_step,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
            )
            results.append(generated[0].strip())
        return results

    # ------------------------------------------------------------------
    # Deduplication & diversity ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / len(sa | sb)

    def _deduplicate(self, candidates: list[str]) -> list[str]:
        unique: list[str] = []
        for c in candidates:
            if c and all(self._jaccard(c, u) < self.dedup_threshold for u in unique):
                unique.append(c)
        return unique

    def _rank_by_diversity(self, texts: list[str]) -> list[str]:
        if len(texts) <= 1:
            return texts
        selected = [texts[0]]
        remaining = texts[1:]
        while remaining:
            best = max(
                remaining,
                key=lambda c: sum(1.0 - self._jaccard(c, s) for s in selected) / len(selected),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, sentence: str, n: int = 5) -> SampleResult:
        if n < 1:
            raise ValueError(f"n must be ≥ 1, got {n}")

        self._load_encoder()
        embedding = self._encode(sentence)       # CPU tensor from here on
        self._load_inverter(embedding.shape[1])  # destroys encoder first

        n_gen = n + max(2, n // 2)
        print(f"[Sampler] Generating {n_gen} candidates for n={n} …")
        raw = self._diverse_generate(embedding, n_gen)

        unique = self._deduplicate(raw)
        final = self._rank_by_diversity(unique)[:n]
        print(f"[Sampler] {len(raw)} generated → {len(unique)} unique → {len(final)} returned")

        return SampleResult(sentence=sentence, embedding=embedding, samples=final)

    def sample_batch(self, sentences: Sequence[str], n: int = 5) -> list[SampleResult]:
        return [self.sample(s, n=n) for s in sentences]

    def unload(self) -> None:
        """Release the inverter from GPU memory."""
        if self._inverter is not None:
            del self._inverter
            self._inverter = None
            _flush_vram()
            print("[Sampler] Inverter unloaded.")
