# Semantic Embedding Reconstruction (ReLoDer)

Inverting dense sentence embeddings back into natural language. Given only a
fixed-size embedding vector — no access to the original text — this project
reconstructs a semantically equivalent sentence.

**Core idea:** a set of small per-position MLPs (`prefix_mlps`) projects a
single sentence embedding into a sequence of soft prefix tokens. Those tokens
are fed as `inputs_embeds` to a LoRA-adapted causal decoder (`Qwen3-0.6B`),
which is trained to regenerate the original text conditioned on nothing but
that prefix.

```
sentence embedding (1024-d)
        │
        ▼
 N × SingleTokenMLP   →  prefix embeddings (N × hidden_dim)
        │
        ▼
 Qwen3-0.6B + LoRA  (inputs_embeds = prefix)
        │
        ▼
 reconstructed text
```

More detail on the architecture and loss: [`docs/architecture.md`](docs/architecture.md).
Dataset field reference: [`docs/dataset_schema.md`](docs/dataset_schema.md).

---

## Repository layout

```
.
├── pyproject.toml      # dependencies (single source of truth)
├── requirements.txt    # exported from uv.lock, for non-uv users
├── uv.lock              # pinned, hashed dependency resolution
├── .env.example          # template for required secrets
│
├── docs/
│   ├── architecture.md
│   └── dataset_schema.md
│
└── src/
    ├── Datasets/
    │   └── dataset_push.py        # builds the embedding dataset, pushes to HF Hub
    ├── Training/
    │   └── gpu_training_script.py # main training entry point
    └── Evaluation/
        └── run_inference.py        # load a checkpoint, generate a reconstruction
```

---

## Datasets

| dataset | size |
|---|---|
| [`jg-eno/MSMACRO-1M-Qwen-Embeddings`](https://huggingface.co/datasets/jg-eno/MSMACRO-1M-Qwen-Embeddings) | 1,000,000 records |
| [`jg-eno/msmarco-v5.1-Qwen-Embeddings`](https://huggingface.co/datasets/jg-eno/msmarco-v5.1-Qwen-Embeddings) | 100,000 records |

Both are built by `src/Datasets/dataset_push.py` from `microsoft/ms_marco`
passages, encoded with `Qwen/Qwen3-Embedding-0.6B`. Field-by-field schema in
[`docs/dataset_schema.md`](docs/dataset_schema.md).

`gpu_training_script.py` currently trains on the 1M dataset
(`MSMACRO-1M-Qwen-Embeddings`).

---

## Model checkpoints

| repo | prefix_len |
|---|---|
| [`jg-eno/ReLoDer_v1`](https://huggingface.co/jg-eno/ReLoDer_v1) | 64 |
| [`jg-eno/ReLoDer_v2`](https://huggingface.co/jg-eno/ReLoDer_v2) | 64 |
| [`jg-eno/ReLoDer_v3`](https://huggingface.co/jg-eno/ReLoDer_v3) | 128 |

Each repo holds two checkpoint files:

- `best_epoch_checkpoint.pt` — best full-epoch average loss
- `best_steps_checkpoint.pt` — best loss over a 20%-of-epoch window

New training runs auto-version: the training script checks the highest
existing `jg-eno/ReLoDer_v{n}` repo and pushes to `v{n+1}`, so re-running
training never overwrites a previous checkpoint.

> **`prefix_len` is not stored in the checkpoint itself** — `run_inference.py`
> auto-detects LoRA rank/target modules from the state dict, but you still
> need to set `prefix_len` by hand to match the table above when loading a
> given version.

---

## Setup

Requires **Python 3.12** and a CUDA-capable GPU for training/dataset
generation (inference can run on CPU, slowly).

```bash
git clone https://github.com/jg-eno/Semantic-Embedding-Reconstruction.git
cd Semantic-Embedding-Reconstruction

curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync                                            # installs the exact pinned versions from uv.lock
source .venv/bin/activate                          # activate the env uv created
```

`uv` is only used here to manage the environment and dependencies. Once the
venv is activated, scripts are run directly with `python`, and notebooks are
run cell by cell in Jupyter (or any notebook frontend pointed at this venv's
kernel) — not via `uv run`.

Without `uv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Secrets

```bash
cp .env.example .env
```

Fill in:
- `HUGGINGFACE_TOKEN` — needs **write** access (pushes datasets/checkpoints to the Hub)
- `WANDB_API_KEY` — training logs to the `MLP_Decoder_Inversion` W&B project

---

## Building the dataset

```bash
python src/Datasets/dataset_push.py
```

Streams `microsoft/ms_marco` (v2.1), encodes passages in batches with
`Qwen3-Embedding-0.6B`, and pushes parquet chunks to the Hub incrementally —
no need to hold the full dataset locally. Target dataset repo is created
**private** by default; change visibility on the Hub if you want others to
load it without a token.

---

## Training

```bash
python src/Training/gpu_training_script.py
```

Key config (edit constants at the top of the script):

| parameter | value |
|---|---|
| decoder | `Qwen/Qwen3-0.6B` |
| dataset | `jg-eno/MSMACRO-1M-Qwen-Embeddings` |
| prefix length | 64 |
| LoRA rank / alpha | 32 / 64 |
| effective batch size | 128 (64 × 2 grad-accum) |
| learning rate | 3e-4 (MLPs), 9e-5 (LoRA) |
| epochs | 20, early stopping patience 3 |

Streams the dataset directly from the Hub, logs to W&B, checkpoints
per-epoch and mid-epoch (every 20% of an epoch) to a local `checkpoints/`
folder, then uploads the best checkpoints to the next auto-versioned
`jg-eno/ReLoDer_v{n}` repo.

---

## Inference

```bash
python src/Evaluation/run_inference.py
```

Edit the `CONFIG` dict at the top of the script to pick a checkpoint —
**make sure `prefix_len` matches the table in
[Model checkpoints](#model-checkpoints)**:

```python
CONFIG = {
    "repo": "jg-eno/ReLoDer_v3",
    "prefix_len": 128,
    "filename": "best_steps_checkpoint.pt",  # or "best_epoch_checkpoint.pt"
}
```