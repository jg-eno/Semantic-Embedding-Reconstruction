# Semantic Embedding Reconstruction (ReLoDer)

Inverting sentence embeddings back into natural language. Given only a dense embedding vector — no access to the original text — this project reconstructs a semantically equivalent sentence.

The core idea: a set of small per-position MLPs (`prefix_mlps`) projects a single sentence embedding into a sequence of soft prefix tokens. Those tokens are fed as `inputs_embeds` to a LoRA-adapted causal decoder (`Qwen3-0.6B`), which is trained to regenerate the original text conditioned on nothing but that prefix.

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

## Repository layout

```
src/
├── Datasets/         # Embedding dataset generation + EDA
├── GPU_Scripts/       # Main training loop and dataset push script (run on GPU)
├── Evaluation/        # ROUGE, cosine-sim, size and embedding-space comparisons
└── HF_Push/            # Hugging Face Hub packaging (config, model, inference)
```

## How it works

**Encoder** — `Qwen/Qwen3-Embedding-0.6B` produces token-level hidden states for a passage, which are mean-pooled over the attention mask and L2-normalized into a single 1024-dim sentence embedding.

**Inverter** — `prefix_len` independent `SingleTokenMLP` modules (`Linear → GELU → Dropout → Linear`) each map that one embedding to one decoder-hidden-dim vector. Stacking their outputs gives a sequence of `prefix_len` synthetic embeddings, which replace normal token embeddings as the decoder's input.

**Decoder** — `Qwen/Qwen3-0.6B` with LoRA adapters (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) is trained to produce the original text autoregressively, conditioned only on the prefix embeddings. The base decoder weights stay frozen; only the LoRA adapters and the prefix MLPs are trained.

**Loss** — a combination of:
- **LM loss** — standard causal cross-entropy on the reconstructed text (prefix positions masked out of the labels).
- **Auxiliary cosine loss** — pushes the mean-pooled prefix embedding toward the mean-pooled target text embedding, keeping the prefix anchored in the same semantic space it came from.

```
loss = lm_loss + AUX_LOSS_WEIGHT * cosine_embedding_loss(mean(prefix), mean(text_embeds))
```

## Dataset

Built from `microsoft/ms_marco` (v1.1 / v2.1) passages, encoded with `Qwen3-Embedding-0.6B` and pushed to the Hub as `jg-eno/msmarco-v5.1-Qwen-Embeddings`. Each record stores:

| field | description |
|---|---|
| `sentence_embeddings` | mean-pooled, L2-normalized passage embedding |
| `token_embeddings` | full per-token hidden states (flattened) |
| `input_ids` / `attention_mask` | tokenized target text |
| `seq_lengths` | true (unpadded) sequence length |
| `texts` | original passage text |

`src/GPU_Scripts/dataset_push.py` builds this dataset by streaming MS MARCO, encoding in batches on GPU, and pushing parquet chunks to the Hub incrementally. `src/Datasets/embedding_generator.py` is an earlier variant of the same pipeline.

## Training

`src/GPU_Scripts/gpu_training_script.py` is the main entry point:

```bash
cd src/GPU_Scripts
pip install -r requirements.txt
python gpu_training_script.py
```

Requires a `.env` file with `HUGGINGFACE_TOKEN` and `WANDB_API_KEY`.

Key configuration (see top of script for the full list):

| parameter | value |
|---|---|
| decoder | `Qwen/Qwen3-0.6B` |
| paragraph dim | 1024 |
| prefix length | 64 |
| LoRA rank / alpha | 32 / 64 |
| effective batch size | 128 (64 × 2 grad-accum) |
| learning rate | 3e-4 (MLPs), 9e-5 (LoRA) |
| scheduler | cosine annealing |
| early stopping | patience 3 epochs |

The script streams the dataset directly from the Hub (no local copy), logs to Weights & Biases, checkpoints both per-epoch and mid-epoch (every 20% of an epoch), and auto-versions its output repo (`jg-eno/ReLoDer_v{n}`) by checking the highest existing version under the namespace before uploading the best checkpoint.


# To-do

- Add Dataset Links, requirements.txt (uv files), Open-Source weights, Running instructions, poetry, instruction about env
- Only 1 requirements.txt, uv.lock and pyproject.toml

- Python 3.12.3