# CLAUDE.md — Sampler Tool

## What this is

A paragraph-embedding inversion tool. Given a sentence, it:
1. Encodes it into a dense embedding via a sentence encoder
2. Passes the embedding through a soft-prefix inverter (ReLoDer) to reconstruct text
3. Runs inference `n` times with escalating temperature to generate diverse outputs
4. Deduplicates near-matches and returns the most spread-out `n` results

The research goal is **semantic embedding reconstruction** — recovering text that is semantically equivalent to (or a plausible variant of) a given sentence, purely from its embedding vector.

---

## Project layout

```
Sampler_Tool/
├── config.py          # All constants — edit this first when changing models/params
├── factory.py         # InverterFactory: downloads checkpoint, auto-detects architecture
├── sampler.py         # Sampler: the main pipeline (encode → invert → deduplicate)
├── app.py             # Flask web UI backend
├── main.py            # CLI entry point
├── models/
│   ├── mlp.py         # SingleTokenMLP — one projection head per prefix position
│   ├── inverter.py    # EndToEndInverter — soft prefix + LoRA decoder
│   └── encoder.py     # Encoder — mean-pooled sentence embeddings
└── templates/
    └── index.html     # Single-page web UI
```

---

## Architecture

### Encoder (`models/encoder.py`)
- Model: `Qwen/Qwen3-Embedding-0.6B`
- Mean-pools token embeddings over non-padding positions, L2-normalises
- Outputs: `(batch, 1024)` fp32 tensor

### Inverter (`models/inverter.py` + `models/mlp.py`)
- **`prefix_len` parallel MLPs** (`SingleTokenMLP`): each projects `(batch, 1024)` → `(batch, decoder_hidden)` through a two-layer network with GELU + Dropout
- These are stacked into a soft prefix of shape `(batch, prefix_len, decoder_hidden)`
- The prefix is fed as `inputs_embeds` to a LoRA-fine-tuned causal LM decoder
- Decoder: `Qwen/Qwen3-0.6B` + LoRA adapters

### Factory (`factory.py`)
All architecture hyperparameters are **auto-detected from the checkpoint** — nothing is hardcoded:
- `prefix_len` — from the max MLP head index in the state dict
- `mlp_hidden_dim` — from `net.0.weight.shape[0]`
- `mlp_dtype` — from the dtype of MLP weights in the checkpoint
- `target_modules`, LoRA `r`, `alpha` — from `lora_A/B` key patterns

### Sampler (`sampler.py`)
Diversity strategy:
```
T_i = T_0 + i × Δ
```
- Sample `i=0` uses base temperature (most faithful)
- Each successive sample gets `+Δ` (more exploratory)
- Generates `n + max(2, n//2)` candidates, deduplicates via Jaccard similarity (threshold 0.85), then ranks remaining candidates by greedy max-dissimilarity

---

## Current checkpoint

| Parameter | Value |
|---|---|
| Repo | `Subhav-K/ReLoDer_v4` |
| File | `best_steps_checkpoint.pt` |
| `prefix_len` | 64 (auto-detected) |
| `mlp_hidden_dim` | 4096 (auto-detected) |
| MLP dtype | `bfloat16` (auto-detected) |
| LoRA `r` | 32 |
| LoRA `alpha` | 64 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

---

## Memory management (RTX 3050 Laptop, 4 GB VRAM)

This GPU has 4 GB total. Fitting both models requires careful sequencing.

### Memory budget

| Component | Size |
|---|---|
| Decoder `Qwen3-0.6B` in fp16 | ~1.1 GB |
| 64 MLP heads, hidden=4096, bf16 | ~0.5 GB |
| Activations + KV cache during generate | ~0.3–0.5 GB |
| **Total** | **~1.9–2.1 GB** |

### Why the encoder cannot stay resident

The encoder (`Qwen3-Embedding-0.6B`) is also ~1.1 GB fp16. Keeping both models on GPU simultaneously would require ~3.2 GB just for weights, leaving almost nothing for activations — causing OOM.

### The two-phase load sequence

```
Phase 1 — Encode
  Encoder loads on GPU (fp16)
  → encode() runs, embedding moved to CPU immediately
  → _flush_vram(): gc.collect() + cuda.empty_cache()
  → encoder.model deleted (not just .to("cpu") — that doesn't release CUDA pages)

Phase 2 — Invert
  Inverter loads:
    decoder via device_map={"": DEVICE} in fp16  → GPU directly
    MLP heads built on CPU, weights loaded, then .to(DEVICE, dtype=bf16)
  → n × generate() calls, all on GPU
```

### Why `.to("cpu")` is not enough

PyTorch's CUDA caching allocator keeps reserved memory pages even after `.to("cpu")`. You need:
```python
del model
gc.collect()
torch.cuda.empty_cache()
```
Only then does the allocator return pages to the OS so the next model can use them.

### dtype decisions

- **Decoder: fp16** — halves weight memory vs fp32 with negligible quality loss for inference
- **MLPs: bfloat16** — matches the dtype they were trained and saved in; avoids a cast that would either waste memory (upcasting to fp32) or lose precision
- **Embeddings: fp32 → cast to bf16 at MLP input** — encoder outputs fp32; `inverter.generate()` casts `paragraph_embs` to the MLP's dtype before the matmul

### Env var set at startup

```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```
Reduces fragmentation on small GPUs where many small allocations follow one large one.

---

## Scaling to a larger GPU

When moving to a GPU with more VRAM (e.g. RTX 3090 24 GB, A100 40/80 GB):

### Parameters to increase

| Parameter | Where | Current (4 GB) | Suggested (24 GB+) |
|---|---|---|---|
| `prefix_len` | `config.py` → `CONFIG["prefix_len"]` (or retrain with more) | 64 | 128–256 |
| Batch size in `Encoder.encode` | pass a list to `sampler.sample_batch` | 1 | 8–32 |
| `MAX_NEW_TOKENS` | `config.py` | 20 | 128–512 |
| `n` samples per call | CLI / UI | 5 | 20–50 |
| MLP `mlp_hidden_dim` | checkpoint-dependent, auto-detected | 4096 | 4096–8192 |

### Things to remove or relax

- The two-phase encode→destroy→invert sequence is unnecessary — both models can stay resident simultaneously
- `torch_dtype=fp16` on the decoder can be dropped; train/infer in bf16 or fp32 for better stability
- The `dedup_threshold=0.85` can be tightened (e.g. 0.7) when generating more candidates at higher n
- `temperature_step` can be reduced (e.g. 0.02) when n is large, to keep the schedule from going too hot

### Code change to keep both models resident (large GPU)

In `sampler.py`, remove `_destroy_encoder()` and simplify `_load_inverter`:

```python
def _load_inverter(self, paragraph_dim: int) -> None:
    if self._inverter is None:
        print("[Sampler] Loading inverter …")
        self._inverter = InverterFactory().load(
            repo=self.cfg["repo"],
            filename=self.cfg["filename"],
            paragraph_dim=paragraph_dim,
        )
        self._tokenizer = self._encoder.tokenizer
```

---

## Running

```bash
# CLI
python main.py
python main.py --sentence "Gravity keeps planets in orbit." --n 8 --temperature 1.5

# Web UI
python app.py
# open http://localhost:5000
```

## Dependencies

Core: `torch`, `transformers`, `peft`, `huggingface_hub`, `flask`

Install: `pip3 install torch transformers peft huggingface_hub flask`
