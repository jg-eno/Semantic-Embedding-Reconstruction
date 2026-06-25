# Dataset schema

Both Hub datasets — `jg-eno/MSMACRO-1M-Qwen-Embeddings` and
`jg-eno/msmarco-v5.1-Qwen-Embeddings` — are produced by the same pipeline
(`src/data/dataset_push.py`) and share the same per-record schema. They
differ only in record count (1M vs 100K) and which pipeline run
produced them.

Source data is `microsoft/ms_marco` (passages field), streamed rather than
downloaded in full, so the script never needs the entire upstream dataset on
disk.

## Fields

| field | dtype (as stored) | shape | description |
|---|---|---|---|
| `sentence_embeddings` | `float32` | `(1024,)` | Mean-pooled, L2-normalized embedding of the full passage, from `Qwen3-Embedding-0.6B`'s `last_hidden_state`. This is the *input* the inverter model consumes. |
| `token_embeddings` | `float32` | `(seq_len * 1024,)`, flattened | Per-token hidden states for the passage, truncated to the true (unpadded) sequence length and flattened to 1-D before storage. Reshape to `(seq_len, 1024)` after loading if you need the original per-token structure. |
| `input_ids` | `int32` | `(seq_len,)` | Tokenized passage text, truncated to the true sequence length (no padding stored). |
| `attention_mask` | `bool` | `(seq_len,)` | Attention mask matching `input_ids`, also truncated to true length. Since padding isn't stored, this is largely redundant in the saved format but kept for compatibility with code expecting it. |
| `seq_lengths` | `int` | scalar | The true, unpadded token count for this passage — what `input_ids`/`attention_mask`/`token_embeddings` were truncated to. |
| `texts` | `string` | scalar | The original passage text, exactly as tokenized. |

## Important: variable-length fields

`input_ids`, `attention_mask`, and `token_embeddings` are **not padded to a
fixed length** in the stored dataset — each record's arrays are exactly
`seq_lengths` long (or `seq_lengths * 1024` for the flattened token
embeddings). Any consumer needs to pad/truncate per-batch at load time. This
is exactly what `prepare_batch()` in `src/training/gpu_training_script.py`
does — it pads every sample up to `MAX_TEXT_LEN` (128) and builds a
matching attention mask, rather than assuming the stored data is already
batch-shaped.

## Storage format

Data is pushed to the Hub as a series of Parquet chunks
(`data/train-chunk{00000..N}.parquet`), each containing `CHUNK_SIZE` (2,048)
records, rather than one single file. This lets the push script run
incrementally against a streamed source dataset without holding the full
dataset in memory, and lets `datasets.load_dataset(..., streaming=True)`
consume it the same way at training time.

## Visibility

Both dataset repos are created with `private=False` by `dataset_push.py`'s
`api.create_repo(...)` call. If you need anonymous/no-token access (e.g. for
others reproducing training), you'll need to switch visibility to public on
the Hub — this isn't something the script does automatically.