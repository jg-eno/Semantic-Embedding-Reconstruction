# Architecture

This is the detailed, module-level companion to the architecture summary in
the main README. Code references point at `src/training/gpu_training_script.py`
and `src/inference/run_inference.py`, which both define equivalent
`SingleTokenMLP` / `EndToEndInverter` classes.

## 1. Encoder

`Qwen/Qwen3-Embedding-0.6B`, frozen, used only to produce the input
embeddings — never trained as part of this pipeline.

```
text → tokenizer → Qwen3-Embedding-0.6B → last_hidden_state (per-token)
                                              │
                                  mean-pool over attention_mask
                                              │
                                       L2-normalize
                                              │
                                  sentence_embedding (1024-d)
```

This same encoding logic is duplicated in `src/data/dataset_push.py` (for
building the training dataset) and in the `Encoder` class inside
`run_inference.py` (for encoding a sentence at inference time). The pooling
and normalization math must match exactly between the two, since the
inverter is trained on embeddings produced by exactly this procedure.

## 2. Inverter (the trained "prefix" component)

```python
class SingleTokenMLP(nn.Module):
    def __init__(self, paragraph_dim, decoder_hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(paragraph_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, decoder_hidden_dim),
        )
```

`prefix_len` independent copies of this module are stacked
(`m_parallel_mlps`), each taking the *same* sentence embedding as input but
learning a distinct projection. Their outputs are stacked along a new
sequence dimension to produce `prefix_len` synthetic token embeddings:

```python
prefix_embeds = torch.stack(
    [mlp(paragraph_embs) for mlp in self.m_parallel_mlps], dim=1
)
# shape: (batch, prefix_len, decoder_hidden_dim)
```

These are the only embeddings the decoder receives as conditioning — there
is no cross-attention or separate encoder-decoder bridge; the prefix is
injected directly into the decoder's normal token-embedding sequence as if
it were a span of real tokens.

## 3. Decoder

`Qwen/Qwen3-0.6B`, loaded in `bfloat16`, with LoRA adapters applied via
`peft`:

```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

The base decoder's weights are frozen by `peft`; only LoRA adapter weights
and the prefix MLPs (`m_parallel_mlps`) receive gradients.

**Forward pass:**

```
prefix_embeds  (batch, prefix_len, hidden)
      +
text_embeds    (batch, seq_len, hidden)   ← real token embeddings of the target text
      =
inputs_embeds  (batch, prefix_len + seq_len, hidden)
      │
      ▼
  decoder(inputs_embeds=..., attention_mask=..., labels=...)
```

Labels for the prefix positions are set to `-100` (ignored by the
cross-entropy loss) — the model is only ever supervised on producing the
*text* tokens, never asked to "predict" the prefix itself.

## 4. Loss

Two terms, summed with a configurable weight:

**LM loss** — standard next-token cross-entropy on the text portion of the
sequence, exactly as in normal causal LM training.

**Auxiliary cosine loss** — encourages the *mean* of the prefix embeddings
to stay close (in cosine distance) to the *mean* of the real text token
embeddings:

```python
mean_prefix = prefix_embeds.mean(dim=1)
mean_text   = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1)
aux_loss    = F.cosine_embedding_loss(mean_prefix, mean_text, target=ones)
```

```
total_loss = lm_loss + AUX_LOSS_WEIGHT * aux_loss
```

The intuition: without this term, the prefix MLPs are free to drift into
any representation that merely happens to help the decoder predict the next
token, with no guarantee it stays anchored in the same embedding space the
input came from. The auxiliary term is a soft constraint pulling the prefix
back toward "the same semantic neighborhood as the target text," which
matters for this project's framing as *semantic* reconstruction rather than
exact verbatim recovery.

## 5. Optimizer / scheduler

Two parameter groups with different learning rates — the prefix MLPs are
trained from scratch (full LR), while the LoRA adapters fine-tune a
pretrained model (reduced LR, ×0.3):

```python
optimizer = AdamW([
    {"params": mlp_params,  "lr": LEARNING_RATE},
    {"params": lora_params, "lr": LEARNING_RATE * 0.3},
], weight_decay=0.01, fused=True)

scheduler = CosineAnnealingLR(optimizer, T_max=..., eta_min=LEARNING_RATE * 0.05)
```

## 6. Checkpointing strategy

Two checkpoints are tracked independently during training:

- **`best_epoch_checkpoint.pt`** — saved when the *full-epoch* average loss
  improves over the previous best (by at least `ES_MIN_DELTA`).
- **`best_steps_checkpoint.pt`** — saved when the average loss over a
  *20%-of-epoch* window improves over the previous best window average.
  This exists because relying on a single batch's loss as a "best step"
  signal would be too noisy; averaging over a fixed window of batches gives
  a more stable comparison point within an epoch, without waiting a full
  epoch to checkpoint.

Early stopping triggers after `ES_PATIENCE` (default 3) consecutive epochs
with no improvement in epoch-level average loss.