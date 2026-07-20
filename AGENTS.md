# AGENTS.md — Semantic Embedding Reconstruction (ReLoDer)

## Project Overview

ReLoDer (**Re**construction via **Lo**RA **Der**coder) inverts dense sentence embeddings back into natural language. Given only a 1024-d embedding vector from a frozen encoder (`Qwen/Qwen3-Embedding-0.6B`), the system reconstructs a semantically equivalent sentence through:

1. **Encoder** — frozen `Qwen/Qwen3-Embedding-0.6B`, mean-pooled + L2-normalized → 1024-d vector
2. **Inverter (prefix MLPs)** — N per-position MLPs project embedding into N soft prefix tokens
3. **Decoder** — LoRA-adapted `Qwen/Qwen3-0.6B` autoregressively regenerates text from prefix tokens

## Tech Stack

- **Python ≥3.12** (uv-managed venv)
- **PyTorch 2.12**, **Transformers 5.12**, **PEFT 0.19** (LoRA)
- **Datasets 5.0**, **HuggingFace Hub 1.20**
- **Weights & Biases** for experiment tracking
- **Flask** (Sampler Tool web UI)
- **NLTK** (sentence tokenization for CNN/DailyMail)

## Commands

### Environment Setup
```bash
uv sync && source .venv/bin/activate   # recommended
# or
pip install -r requirements.txt
cp .env.example .env  # fill in HUGGINGFACE_TOKEN + WANDB_API_KEY
```

### Dataset Building
```bash
python src/Datasets/dataset_push.py          # MS MARCO → HF Hub
python src/Datasets/cnn_dataset_push.py      # CNN/DailyMail → HF Hub
python src/Datasets/eda_analysis.py          # Token length EDA
```

### Training
```bash
python src/Training/gpu_training_script.py   # Full training pipeline
```
Key defaults: prefix_len=128, LoRA r=32 α=64, batch=64×2grad_accum, MLP LR=3e-4, LoRA LR=9e-5, epochs=10, early stopping patience=3.

### Inference
```bash
python src/Evaluation/run_inference.py       # Edit CONFIG dict at top
```

### Hyperparameter Sweep (W&B)
```bash
cd src/Abalation_Studies/HyperParam_Sweep
wandb sweep sweep_config.yaml && wandb agent <sweep_id>
```

### Sampler Tool
```bash
python src/Sampler_Tool/main.py --sentence "..." --n 5
python src/Sampler_Tool/app.py               # Web UI on :5000
```

## Code Organization

```
src/
├── Datasets/          # Dataset building & EDA scripts
├── Training/          # gpu_training_script.py (main training entry)
├── Evaluation/        # run_inference.py, eval-script.ipynb
├── Sampler_Tool/      # Standalone diverse-reconstruction tool
│   ├── models/        # SingleTokenMLP, EndToEndInverter, Encoder
│   ├── config.py      # All constants and defaults
│   ├── factory.py     # Checkpoint download + architecture auto-detection
│   ├── sampler.py     # encode → invert → deduplicate → diversity rank
│   ├── app.py         # Flask web UI backend
│   ├── main.py        # CLI entry point
│   └── templates/     # HTML for web UI
└── Abalation_Studies/ # Hyperparameter sweep scripts + configs
```

## Key Models

| Class | Location | Purpose |
|---|---|---|
| `Encoder` | `src/Sampler_Tool/models/encoder.py` | Mean-pooled L2-normalized sentence embeddings |
| `SingleTokenMLP` | `src/Sampler_Tool/models/mlp.py` | Linear→GELU→Dropout→Linear, one per prefix position |
| `EndToEndInverter` | `src/Sampler_Tool/models/inverter.py` | Stacks MLPs into prefix, calls decoder.generate() |

Note: These model classes are duplicated in `gpu_training_script.py` and `run_inference.py` with minor variations. The Sampler Tool version (`src/Sampler_Tool/models/`) is the cleanest.

## Datasets (Published)

| Dataset | Records | Description |
|---|---|---|
| `jg-eno/MSMACRO-1M-Qwen-Embeddings` | 1M | MS MARCO passages, Qwen3-Embedding-0.6B encoded |
| `jg-eno/cnn-dailymail-chunked-512-embeddings` | Variable | CNN/DailyMail, sentence-tokenized, ≤512 tokens |

Schema: `sentence_embeddings`, `token_embeddings`, `input_ids`, `attention_mask`, `seq_lengths`, `texts`.

## Checkpoints

| Repo | prefix_len |
|---|---|
| `jg-eno/ReLoDer_v1` | 64 |
| `jg-eno/ReLoDer_v2` | 64 |
| `jg-eno/ReLoDer_v3` | 128 |
| `Subhav-K/ReLoDer_v4` | 64 |

## Lint / Typecheck / Tests

**No automated tests, linting, or type-checking are configured.** The `.gitignore` references `ruff_cache` so ruff may have been used previously; consider adding `ruff check src/` if setting up linting.

## Conventions

- Hyperparameters and config are defined as **module-level constants** at the top of scripts (not CLI args), except the Sampler Tool which uses argparse.
- Checkpoints are `.pt` files containing `{"model_state_dict", "optimizer_state_dict", "epoch", "steps", "config"}`.
- All model artifacts are pushed to HuggingFace Hub. Use `HUGGINGFACE_TOKEN` from `.env`.
- Experiments are logged to Weights & Biases. Use `WANDB_API_KEY` from `.env`.
- The codebase does not follow a strict Python package layout (`src/` has no top-level `__init__.py`).
