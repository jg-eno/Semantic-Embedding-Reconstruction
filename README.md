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

---

## Repository layout

```
.
├── README.md
├── pyproject.toml          # single source of truth for dependencies
├── requirements.txt        # exported from uv.lock, for non-uv users
├── uv.lock                 # pinned, hashed resolution of pyproject.toml
├── .env.example             # template for required secrets
├── .gitignore
│
├── docs/
│   ├── architecture.md      # encoder / inverter / decoder / loss, in detail
│   └── dataset_schema.md    # field-by-field description of the HF datasets
│
└── src/
    ├── data/
    │   ├── dataset_push.py    # streams MS MARCO, encodes, pushes to HF Hub
    │   └── max_token_len.py   # one-off EDA helper over a local .h5 export
    │
    ├── training/
    │   └── gpu_training_script.py   # main training entry point
    │
    ├── inference/
    │   └── run_inference.py    # load a checkpoint from the Hub, generate text
    │
    └── evaluation/
        ├── run_inference.py    # (evaluation-oriented copy / shared logic — see note below)
        └── eval-script_.ipynb   # ROUGE, cosine-sim, embedding-space comparisons
```
---

## How it works

**Encoder** — `Qwen/Qwen3-Embedding-0.6B` produces token-level hidden states
for a passage, which are mean-pooled over the attention mask and L2-normalized
into a single 1024-dim sentence embedding.

**Inverter** — `prefix_len` independent `SingleTokenMLP` modules
(`Linear → GELU → Dropout → Linear`) each map that one embedding to one
decoder-hidden-dim vector. Stacking their outputs gives a sequence of
`prefix_len` synthetic embeddings, which replace normal token embeddings as
the decoder's input.

**Decoder** — `Qwen/Qwen3-0.6B` with LoRA adapters
(`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
is trained to produce the original text autoregressively, conditioned only on
the prefix embeddings. The base decoder weights stay frozen; only the LoRA
adapters and the prefix MLPs are trained.

**Loss** — a combination of:

- **LM loss** — standard causal cross-entropy on the reconstructed text
  (prefix positions masked out of the labels).
- **Auxiliary cosine loss** — pushes the mean-pooled prefix embedding toward
  the mean-pooled target text embedding, keeping the prefix anchored in the
  same semantic space it came from.

```
loss = lm_loss + AUX_LOSS_WEIGHT * cosine_embedding_loss(mean(prefix), mean(text_embeds))
```

See [`docs/architecture.md`](docs/architecture.md) for the full module-level
breakdown.

---

## Datasets

Built from `microsoft/ms_marco` passages, encoded with `Qwen3-Embedding-0.6B`.
Two versions exist on the Hub, at different scales:

| dataset | size | built by |
|---|---|---|
| [`jg-eno/MSMACRO-1M-Qwen-Embeddings`](https://huggingface.co/datasets/jg-eno/MSMACRO-1M-Qwen-Embeddings) | 1,000,000 records | `src/data/dataset_push.py` |
| [`jg-eno/msmarco-v5.1-Qwen-Embeddings`](https://huggingface.co/datasets/jg-eno/msmarco-v5.1-Qwen-Embeddings) | 100,000 records | earlier pipeline run |

---

## Model checkpoints

| repo | notes |
|---|---|
| [`jg-eno/ReLoDer_v1`](https://huggingface.co/jg-eno/ReLoDer_v1) | earlier run |
| [`jg-eno/ReLoDer_v2`](https://huggingface.co/jg-eno/ReLoDer_v2) | earlier run |
| [`jg-eno/ReLoDer_v3`](https://huggingface.co/jg-eno/ReLoDer_v3) | current default in `run_inference.py` |

Each repo contains two checkpoint files (see `gpu_training_script.py`'s
upload step):

- `best_epoch_checkpoint.pt` — best full-epoch average loss
- `best_steps_checkpoint.pt` — best loss over a 20%-of-epoch window (finer-grained than per-epoch)

New training runs are **auto-versioned**: `gpu_training_script.py` checks the
highest existing `jg-eno/ReLoDer_v{n}` repo under the namespace at startup
and pushes to `v{n+1}`, so re-running training never overwrites a previous
checkpoint.

> `run_inference.py` recovers LoRA `r` / `target_modules` automatically from
> the checkpoint's state dict, but `prefix_len` is *not* auto-detected — you
> must set it correctly per checkpoint in the `CONFIG` dict.

---

## Setup

Requires **Python 3.12** and a CUDA-capable GPU for training/dataset
generation (inference can run on CPU, slowly). Dependency management is via
[`uv`](https://docs.astral.sh/uv/).

```bash
# Clone
git clone https://github.com/jg-eno/Semantic-Embedding-Reconstruction.git
cd Semantic-Embedding-Reconstruction

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the venv and install the exact pinned dependency set
uv sync
```

`uv sync` reads `uv.lock` and installs the exact pinned, hashed dependency
versions — this is what guarantees reproducibility across machines, rather
than re-resolving versions from scratch.

If you're not using `uv`, a plain `requirements.txt` is also provided:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Secrets

Both the dataset-building and training scripts load credentials via
`python-dotenv`. Copy the template and fill in your own keys:

```bash
cp .env.example .env
```

You need:

- `HUGGINGFACE_TOKEN` — a Hugging Face token with **write** access (to push
  dataset chunks / model checkpoints to the Hub)
- `WANDB_API_KEY` — a Weights & Biases API key (training logs to the
  `MLP_Decoder_Inversion` W&B project)

### GPU / CUDA notes

`pyproject.toml` pins a CUDA-enabled `torch==2.12.1` build along with the
matching `nvidia-*` runtime packages (resolved automatically as transitive
dependencies — you don't list them yourself). If `uv sync` resolves a
CPU-only `torch` on your machine, check that your platform/CUDA toolchain
matches what's expected by the lockfile, or re-resolve with an explicit
PyTorch index.

---

## Building the dataset

`src/data/dataset_push.py` streams `microsoft/ms_marco` (v2.1), encodes
passages in batches with `Qwen3-Embedding-0.6B`, and pushes parquet chunks to
the Hub incrementally (so you never need the full dataset on local disk at
once).

```bash
uv run src/data/dataset_push.py
```

Key constants at the top of the script:

| constant | value | meaning |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | encoder used to generate embeddings |
| `BATCH_SIZE` | 128 | passages encoded per forward pass |
| `DATASET_NAME` | `jg-eno/msmacro-1M-Qwen-Embeddings` | target HF dataset repo |
| `MAX_SAMPLES` | 1,000,000 | total passages to process |
| `CHUNK_SIZE` | 2,000 | samples per pushed parquet chunk |

The target dataset repo is created **private** by default
(`private=True` in `api.create_repo`) — make it public on the Hub yourself
if you want others to load it without authenticating.

`src/data/max_token_len.py` is a small standalone EDA helper — it expects a
local `.h5` export (not produced by `dataset_push.py` itself) and just prints
the average sequence length. Treat it as a debugging scratch script rather
than part of the main pipeline.

---

## Training

```bash
uv run src/training/gpu_training_script.py
```

The script streams the dataset directly from the Hub (no local copy
required), logs metrics to Weights & Biases, checkpoints both per-epoch and
mid-epoch (every 20% of an epoch), and auto-versions its output repo as
described in [Model checkpoints](#model-checkpoints).

Key configuration (edit these constants at the top of the script):

| parameter | value | meaning |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-0.6B` | decoder base model |
| `DATASET` | `jg-eno/msmarco-v5.1-Qwen-Embeddings` | training data (100k records) |
| `PARAGRAPH_DIM` | 1024 | input embedding dimension |
| `PREFIX_LEN` | 64 | number of soft prefix tokens |
| `MAX_TEXT_LEN` | 128 | max tokens of target text per sample |
| `BATCH_SIZE` / `GRAD_ACCUM` | 64 / 2 | effective batch size = 128 |
| `LEARNING_RATE` | 3e-4 (MLPs), 0.3× that for LoRA params | separate LR per param group |
| LoRA `r` / `alpha` | 32 / 64 | adapter rank / scaling |
| `NUM_EPOCHS` | 20 | with early stopping, patience 3 |
| scheduler | cosine annealing | over the full planned run |

Local checkpoints land in `checkpoints/best_epoch_checkpoint.pt` and
`checkpoints/best_steps_checkpoint.pt` (directory is gitignored — `uv run`
creates it automatically on first run). At the end of training, both files
are uploaded to the next auto-versioned `jg-eno/ReLoDer_v{n}` Hub repo.

**To reproduce a specific past run exactly,** match `PREFIX_LEN`, LoRA
`r`/`alpha`, and `DATASET` to whatever that checkpoint's README/model card
states — these have varied between `v1`/`v2`/`v3` (see
[Known inconsistencies](#known-inconsistencies--things-to-watch)).

---

## Inference

`src/inference/run_inference.py` downloads a checkpoint from the Hub, infers
its LoRA configuration directly from the checkpoint's state dict (no need to
hardcode `r`/`target_modules` per version), encodes a sample sentence, and
generates a reconstruction.

```bash
uv run src/inference/run_inference.py
```

Edit the `CONFIG` dict at the top of the script to point at a different
checkpoint:

```python
CONFIG = {
    "repo": "jg-eno/ReLoDer_v3",
    "prefix_len": 128,                      # must match how this checkpoint was trained
    "filename": "best_steps_checkpoint.pt",  # or "best_epoch_checkpoint.pt"
}
```

`prefix_len` is **not** recoverable from the checkpoint automatically —
LoRA rank and target modules are detected from tensor shapes, but the prefix
MLP count isn't stored anywhere in the state dict's key names, so you must
set it yourself to match the training run that produced the checkpoint.

---

## Evaluation

`src/evaluation/eval-script.ipynb` runs the fuller evaluation suite: ROUGE
scores, cosine similarity between original and reconstructed-text
embeddings, and embedding-space comparisons across checkpoint versions.

```bash
uv run --with jupyter jupyter lab src/evaluation/eval-script.ipynb
```

(or open it in your preferred notebook environment with the project's `uv`
environment selected as the kernel).

---