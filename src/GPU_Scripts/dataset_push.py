from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from huggingface_hub import login
from dotenv import load_dotenv
import os

# ==========================================
# AUTHENTICATION
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HF_TOKEN)

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # change this
BATCH_SIZE = 128   # MUCH larger for GPU utilization
DATASET_NAME = "jg-eno/msmacro-1M-Qwen-Embeddings"
MAX_LENGTH = 512
MAX_SAMPLES = 1_000_000
CHUNK_SIZE = 2000  # push every N samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(f"Device: {DEVICE}")

# ---------------- LOAD ----------------
dataset = load_dataset(
    "microsoft/ms_marco",
    "v2.1",
    split="train",
    streaming=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model) 
model.eval()
from huggingface_hub import HfApi
import io

api = HfApi()
api.create_repo(
    repo_id=DATASET_NAME,
    repo_type="dataset",
    private=True,
    exist_ok=True
)

# ---------------- BUFFER ----------------
buffer = {
    "token_embeddings": [],
    "sentence_embeddings": [],
    "input_ids": [],
    "attention_mask": [],
    "seq_lengths": [],
    "texts": [],
}

sample_count = 0
chunk_id = 0
global_batch = []

def push_buffer(buffer, chunk_id):
    ds = Dataset.from_dict(buffer)
    parquet_buffer = io.BytesIO()
    ds.to_parquet(parquet_buffer)
    parquet_buffer.seek(0)
    api.upload_file(
        path_or_fileobj=parquet_buffer,
        path_in_repo=f"data/train-chunk{chunk_id:05d}.parquet",
        repo_id=DATASET_NAME,
        repo_type="dataset",
        commit_message=f"chunk {chunk_id}"
    )
    print(f"Pushed chunk {chunk_id} ({len(buffer['texts'])} samples)")

# ---------------- MAIN LOOP ----------------
print("Processing & pushing to HF...")

for example in tqdm(dataset):

    passages = example["passages"]["passage_text"]
    global_batch.extend(passages)

    # Process when batch full
    while len(global_batch) >= BATCH_SIZE:

        batch_texts = global_batch[:BATCH_SIZE]
        global_batch = global_batch[BATCH_SIZE:]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        # Faster transfer
        inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask   = inputs["attention_mask"]
        input_ids        = inputs["input_ids"]

        # Sentence embeddings
        mask_expanded       = attention_mask.unsqueeze(-1).float()
        masked              = token_embeddings * mask_expanded
        sentence_embeddings = masked.sum(dim=1) / mask_expanded.sum(dim=1)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Move to CPU once
        token_embeddings_np    = token_embeddings.cpu().numpy().astype("float32")
        sentence_embeddings_np = sentence_embeddings.cpu().numpy().astype("float32")
        attention_mask_np      = attention_mask.cpu().numpy().astype(bool)
        input_ids_np           = input_ids.cpu().numpy().astype(np.int32)

        # Flatten per sample
        for i in range(len(batch_texts)):
            seq_len = int(attention_mask_np[i].sum())

            buffer["token_embeddings"].append(token_embeddings_np[i][:seq_len].flatten())
            buffer["sentence_embeddings"].append(sentence_embeddings_np[i])
            buffer["input_ids"].append(input_ids_np[i][:seq_len])
            buffer["attention_mask"].append(attention_mask_np[i][:seq_len])
            buffer["seq_lengths"].append(seq_len)
            buffer["texts"].append(batch_texts[i])

        sample_count += len(batch_texts)

        # ---------------- PUSH CHUNK ----------------
        if len(buffer["texts"]) >= CHUNK_SIZE:
            push_buffer(buffer, chunk_id)
            buffer = {k: [] for k in buffer}
            chunk_id += 1

        if MAX_SAMPLES and sample_count >= MAX_SAMPLES:
            break

    if MAX_SAMPLES and sample_count >= MAX_SAMPLES:
        break


# ---------------- FINAL PUSH ----------------
if len(buffer["texts"]) > 0:
    push_buffer(buffer, chunk_id)

print(f"Done. Total samples: {sample_count}")