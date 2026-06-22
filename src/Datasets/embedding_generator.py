from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

# ---------------- CONFIG ----------------
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DATASET_NAME = "jg-eno/msmarco-v1.1-Qwen-Embeddings"  # change this
BATCH_SIZE = 8   # MUCH larger for GPU utilization
MAX_LENGTH = 512
MAX_SAMPLES = 100000
CHUNK_SIZE = 2000  # push every N samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

print(f"Device: {DEVICE}")

# ---------------- LOAD ----------------
dataset = load_dataset(
    "microsoft/ms_marco",
    "v1.1",
    split="train",
    streaming=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

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

        # Sentence embeddings (IDENTICAL LOGIC)
        mask_expanded       = attention_mask.unsqueeze(-1).float()
        masked              = token_embeddings * mask_expanded
        sentence_embeddings = masked.sum(dim=1) / mask_expanded.sum(dim=1)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Move to CPU once
        token_embeddings_np   = token_embeddings.cpu().numpy().astype("float32")
        sentence_embeddings_np = sentence_embeddings.cpu().numpy().astype("float32")
        attention_mask_np     = attention_mask.cpu().numpy().astype(bool)
        input_ids_np          = input_ids.cpu().numpy().astype(np.int32)

        # Flatten per sample (same as your logic)
        for i in range(len(batch_texts)):
            seq_len = int(attention_mask_np[i].sum())

            buffer["token_embeddings"].append(
                token_embeddings_np[i][:seq_len].flatten()
            )

            buffer["sentence_embeddings"].append(
                sentence_embeddings_np[i]
            )

            buffer["input_ids"].append(
                input_ids_np[i][:seq_len]
            )

            buffer["attention_mask"].append(
                attention_mask_np[i][:seq_len]
            )

            buffer["seq_lengths"].append(seq_len)
            buffer["texts"].append(batch_texts[i])

        sample_count += len(batch_texts)

        # ---------------- PUSH CHUNK ----------------
        if len(buffer["texts"]) >= CHUNK_SIZE:

            ds = Dataset.from_dict(buffer)

            ds.push_to_hub(
                DATASET_NAME,
                split="train",
                private=True,
                commit_message=f"chunk {chunk_id}"
            )

            print(f"Pushed chunk {chunk_id} ({len(buffer['texts'])} samples)")

            buffer = {k: [] for k in buffer}
            chunk_id += 1

        if MAX_SAMPLES and sample_count >= MAX_SAMPLES:
            break

    if MAX_SAMPLES and sample_count >= MAX_SAMPLES:
        break


# ---------------- FINAL PUSH ----------------
if len(buffer["texts"]) > 0:
    ds = Dataset.from_dict(buffer)
    ds.push_to_hub(
        DATASET_NAME,
        split="train",
        private=True,
        commit_message=f"final chunk {chunk_id}"
    )

print(f"Done. Total samples: {sample_count}")