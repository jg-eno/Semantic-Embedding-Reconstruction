"""
Build sentence-chunked records from abisee/cnn_dailymail, generate token &
sentence embeddings, then push to HF as chunked parquet files.

Chunking strategy
-----------------
Each article is split into sentences (NLTK Punkt).  Consecutive sentences are
greedily packed into chunks that stay <= MAX_LENGTH tokens, strictly within a
single article.  Every chunk is therefore a topically coherent span of one
story, naturally close to 512 tokens with near-zero padding.
"""

import io
import os

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ==========================================
# AUTHENTICATION
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HF_TOKEN)

# Ensure the Punkt sentence-tokenizer data is present.
nltk.download("punkt_tab", quiet=True)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ==========================================
# CONFIG
# ==========================================
MODEL_NAME   = "Qwen/Qwen3-Embedding-0.6B"
DATASET_NAME = "abisee/cnn_dailymail"
SUBSET       = "3.0.0"
SPLITS       = ["train", "validation", "test"]
HF_REPO      = "jg-eno/cnn-dailymail-chunked-512-embeddings"

MAX_LENGTH   = 512      # max tokens per chunk
BATCH_SIZE   = 64      # sentences fed to the model at once
CHUNK_SIZE   = 2000     # push a parquet file every N samples
MAX_SAMPLES  = 1_000_000     # set to an int to cap early, None = full dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ==========================================
# MODEL & TOKENIZER
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.eval()

# ==========================================
# HF REPO
# ==========================================
api = HfApi()
api.create_repo(repo_id=HF_REPO, repo_type="dataset", exist_ok=True)

# ==========================================
# HELPERS
# ==========================================

def _token_len(text: str) -> int:
    """Token count for *text* without special tokens."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _chunk_article(article: str) -> list[str]:
    """
    Split one article into sentences, then greedily pack consecutive sentences
    into chunks <= MAX_LENGTH tokens.  Never crosses into another article.
    A single sentence longer than MAX_LENGTH becomes its own oversized chunk.
    """
    sentences = nltk.sent_tokenize(article)
    chunks: list[str] = []
    buffer_sents: list[str] = []
    buffer_len = 0

    for sent in sentences:
        sent_len = _token_len(sent)
        if buffer_len + sent_len <= MAX_LENGTH:
            buffer_sents.append(sent)
            buffer_len += sent_len
        else:
            if buffer_sents:
                chunks.append(" ".join(buffer_sents))
            buffer_sents = [sent]
            buffer_len   = sent_len

    if buffer_sents:
        chunks.append(" ".join(buffer_sents))

    return chunks


def push_buffer(buffer: dict, chunk_id: int) -> None:
    ds = Dataset.from_dict(buffer)
    parquet_buffer = io.BytesIO()
    ds.to_parquet(parquet_buffer)
    parquet_buffer.seek(0)
    api.upload_file(
        path_or_fileobj=parquet_buffer,
        path_in_repo=f"data/train-chunk{chunk_id:05d}.parquet",
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=f"chunk {chunk_id}",
    )
    print(f"Pushed chunk {chunk_id} ({len(buffer['texts'])} samples)")


def process_batch(batch_texts: list[str]) -> dict:
    """
    Tokenize *batch_texts*, run the model, and return a dict of per-sample
    numpy arrays ready to extend the buffer.
    """
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state   # (B, T, H)
    attention_mask   = inputs["attention_mask"]    # (B, T)
    input_ids        = inputs["input_ids"]         # (B, T)

    # Masked mean pooling → sentence embedding
    mask_expanded       = attention_mask.unsqueeze(-1).float()
    masked              = token_embeddings * mask_expanded
    sentence_embeddings = masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # Move to CPU once
    tok_emb_np  = token_embeddings.cpu().numpy().astype("float32")
    sent_emb_np = sentence_embeddings.cpu().numpy().astype("float32")
    mask_np     = attention_mask.cpu().numpy().astype(bool)
    ids_np      = input_ids.cpu().numpy().astype(np.int32)

    result: dict = {
        "token_embeddings":    [],
        "sentence_embeddings": [],
        "input_ids":           [],
        "attention_mask":      [],
        "seq_lengths":         [],
        "texts":               [],
    }

    for i in range(len(batch_texts)):
        seq_len = int(mask_np[i].sum())
        result["token_embeddings"].append(tok_emb_np[i][:seq_len].flatten())
        result["sentence_embeddings"].append(sent_emb_np[i])
        result["input_ids"].append(ids_np[i][:seq_len])
        result["attention_mask"].append(mask_np[i][:seq_len])
        result["seq_lengths"].append(seq_len)
        result["texts"].append(batch_texts[i])

    return result


# ==========================================
# MAIN LOOP
# ==========================================
buffer: dict = {
    "token_embeddings":    [],
    "sentence_embeddings": [],
    "input_ids":           [],
    "attention_mask":      [],
    "seq_lengths":         [],
    "texts":               [],
}

sample_count  = 0
chunk_id      = 0
global_batch: list[str] = []   # accumulates chunk texts across articles

print("Processing & pushing to HF...")

for split in SPLITS:
    dataset = load_dataset(DATASET_NAME, name=SUBSET, split=split, streaming=True)

    for example in tqdm(dataset, desc=split):
        article = example["article"].strip()
        if not article:
            continue

        # Break article into sentence-bounded chunks and queue them.
        global_batch.extend(_chunk_article(article))

        # Drain the batch queue in BATCH_SIZE steps.
        while len(global_batch) >= BATCH_SIZE:
            batch_texts  = global_batch[:BATCH_SIZE]
            global_batch = global_batch[BATCH_SIZE:]

            result = process_batch(batch_texts)
            for key in buffer:
                buffer[key].extend(result[key])

            sample_count += len(batch_texts)

            if len(buffer["texts"]) >= CHUNK_SIZE:
                push_buffer(buffer, chunk_id)
                buffer   = {k: [] for k in buffer}
                chunk_id += 1

            if MAX_SAMPLES and sample_count >= MAX_SAMPLES:
                break

        if MAX_SAMPLES and sample_count >= MAX_SAMPLES:
            break

    if MAX_SAMPLES and sample_count >= MAX_SAMPLES:
        break

# Process any leftover texts that didn't fill a full batch.
if global_batch:
    result = process_batch(global_batch)
    for key in buffer:
        buffer[key].extend(result[key])
    sample_count += len(global_batch)

# Final push for whatever remains in the buffer.
if buffer["texts"]:
    push_buffer(buffer, chunk_id)

print(f"Done. Total samples: {sample_count}")
