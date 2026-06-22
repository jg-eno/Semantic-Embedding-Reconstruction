import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from huggingface_hub import login, HfApi
from pathlib import Path
from dotenv import load_dotenv
import os

# ==========================================
# AUTHENTICATION
# ==========================================
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_KEY = os.getenv("WANDB_API_KEY")
login(token=HF_TOKEN)
wandb.login(key=WANDB_KEY)

# ==========================================
# AUTO-VERSIONED REPO NAME
# ==========================================
import re


def resolve_next_repo_version(prefix: str) -> str:
    """
    Looks up existing model repos under the user's namespace matching
    `prefix{int}`, finds the highest existing version, and returns
    `prefix{highest + 1}`. Falls back to `prefix1` if none exist yet.

    `prefix` is expected to look like "jg-eno/ReLoDer_v" (namespace + base
    name, version number omitted).
    """
    namespace, _, base_name = prefix.rpartition("/")
    pattern = re.compile(rf"^{re.escape(base_name)}(\d+)$")

    api = HfApi()
    try:
        existing_models = api.list_models(author=namespace)
    except Exception as e:
        print(f"  Warning: could not list existing repos ({e}). Defaulting to version 1.")
        return f"{prefix}1"

    highest_version = 0
    for m in existing_models:
        repo_name = m.id.split("/")[-1]
        match = pattern.match(repo_name)
        if match:
            highest_version = max(highest_version, int(match.group(1)))

    next_version = highest_version + 1
    resolved = f"{prefix}{next_version}"
    print(f"  Auto-versioned repo: highest existing = v{highest_version} -> using {resolved}")
    return resolved


REPO_PREFIX     = "jg-eno/ReLoDer_v"   # auto-versioned: actual repo resolved at runtime as REPO_PREFIX + next int
TARGET_REPO = resolve_next_repo_version(REPO_PREFIX)
print(f"  Target HF repo for this run: {TARGET_REPO}\n")

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME      = "Qwen/Qwen3-0.6B"
PARAGRAPH_DIM   = 1024
PREFIX_LEN      = 64
MAX_TEXT_LEN    = 128
BATCH_SIZE      = 64
GRAD_ACCUM      = 2          # Effective batch = 128
LEARNING_RATE   = 3e-4
TOTAL_SAMPLES   = 100_000   # int, not float — avoids float leaking into batch/scheduler math
NUM_EPOCHS      = 20
BEST_EPOCH_PATH = Path("checkpoints/best_epoch_checkpoint.pt")
BEST_STEPS_PATH = Path("checkpoints/best_steps_checkpoint.pt")
DATASET         = "jg-eno/msmarco-v5.1-Qwen-Embeddings"
LOG_INTERVAL    = 50         # Print loss to terminal every N batches
AUX_LOSS_WEIGHT = 1        # Weight on cosine aux loss relative to LM loss — tune this

# Early stopping
ES_PATIENCE     = 3          # Stop if no improvement for this many epochs
ES_MIN_DELTA    = 1e-4       # Minimum improvement to count as progress

BATCHES_PER_EPOCH   = TOTAL_SAMPLES // BATCH_SIZE
CHECKPOINT_INTERVAL = int(BATCHES_PER_EPOCH * 0.20)

BEST_EPOCH_PATH.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

# ==========================================
# WANDB
# ==========================================
wandb.init(
    project="MLP_Decoder_Inversion",
    config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "prefix_len": PREFIX_LEN,
        "max_text_len": MAX_TEXT_LEN,
        "num_epochs": NUM_EPOCHS,
        "es_patience": ES_PATIENCE,
        "target_repo": TARGET_REPO,
    },
)

# ==========================================
# MODEL
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base_decoder = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": device},
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",  # MLP layers — worth ablating if recon quality plateaus
    ],
)
decoder_with_lora = get_peft_model(base_decoder, lora_config)


class SingleTokenMLP(nn.Module):
    def __init__(self, paragraph_dim: int, decoder_hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(paragraph_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, decoder_hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EndToEndInverter(nn.Module):
    def __init__(
        self,
        paragraph_dim: int,
        decoder_hidden_dim: int,
        prefix_len: int,
        decoder_model: nn.Module,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.decoder_hidden_dim = decoder_hidden_dim
        self.m_parallel_mlps = nn.ModuleList(
            [SingleTokenMLP(paragraph_dim, decoder_hidden_dim) for _ in range(prefix_len)]
        )
        self.decoder = decoder_model

    def forward(
        self,
        paragraph_embs: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = paragraph_embs.shape[0]
        device     = paragraph_embs.device

        prefix_embeds = torch.stack(
            [mlp(paragraph_embs) for mlp in self.m_parallel_mlps], dim=1
        )

        text_embeds   = self.decoder.get_input_embeddings()(text_input_ids)
        prefix_embeds = prefix_embeds.to(dtype=text_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

        prefix_labels = torch.full(
            (batch_size, self.prefix_len), -100, dtype=torch.long, device=device
        )
        labels = torch.cat([prefix_labels, text_input_ids], dim=1)

        prefix_mask = torch.ones(
            (batch_size, self.prefix_len), dtype=attention_mask.dtype, device=device
        )
        concat_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        lm_loss = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=concat_mask,
            labels=labels,
        ).loss

        mean_prefix = prefix_embeds.mean(dim=1)
        mask_exp    = attention_mask.unsqueeze(-1).to(dtype=text_embeds.dtype)
        mean_text   = (text_embeds * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        target      = torch.ones(batch_size, device=device)
        aux_loss    = nn.functional.cosine_embedding_loss(
            mean_prefix.float(), mean_text.float(), target
        )

        return lm_loss, aux_loss


model = EndToEndInverter(
    paragraph_dim=PARAGRAPH_DIM,
    decoder_hidden_dim=base_decoder.config.hidden_size,
    prefix_len=PREFIX_LEN,
    decoder_model=decoder_with_lora,
)
model.m_parallel_mlps.to(device=device, dtype=torch.bfloat16)

# ==========================================
# OPTIMIZER & SCHEDULER
# ==========================================
# Separate LR for the from-scratch prefix MLPs vs. the pretrained LoRA adapters.
# The MLPs are randomly initialized and need a stronger signal; the LoRA
# adapters sit on top of pretrained weights and tend to destabilize at the
# same LR that's appropriate for a from-scratch module.
mlp_params  = list(model.m_parallel_mlps.parameters())
lora_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad]

optimizer = AdamW(
    [
        {"params": mlp_params,  "lr": LEARNING_RATE},
        {"params": lora_params, "lr": LEARNING_RATE * 0.3},
    ],
    weight_decay=0.01,
    fused=True,
)

# T_max spans the full training run across all epochs
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=(BATCHES_PER_EPOCH * NUM_EPOCHS) // GRAD_ACCUM,
    eta_min=LEARNING_RATE * 0.05,
)

# ==========================================
# DATA UTILITIES
# ==========================================
def prepare_batch(batch_samples: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    paragraph_embs_list, input_ids_list, attention_mask_list = [], [], []

    for sample in batch_samples:
        sent_emb = torch.tensor(sample["sentence_embeddings"], dtype=torch.float32)
        paragraph_embs_list.append(sent_emb)

        ids      = sample["input_ids"]
        seq_len  = min(len(ids), MAX_TEXT_LEN)

        padded_ids           = np.full(MAX_TEXT_LEN, tokenizer.pad_token_id, dtype=np.int64)
        padded_ids[:seq_len] = ids[:seq_len]
        input_ids_list.append(torch.tensor(padded_ids))

        mask           = np.zeros(MAX_TEXT_LEN, dtype=np.float32)
        mask[:seq_len] = 1.0
        attention_mask_list.append(torch.tensor(mask))

    return (
        torch.stack(paragraph_embs_list),
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
    )

# ==========================================
# TRAINING LOOP
# ==========================================
print(f"Effective batch size : {BATCH_SIZE * GRAD_ACCUM}")
print(f"Epochs               : {NUM_EPOCHS}")
print(f"Early stopping       : patience={ES_PATIENCE}, min_delta={ES_MIN_DELTA}")
print(f"Terminal log every   : {LOG_INTERVAL} batches")
print(f"Checkpoint every     : {CHECKPOINT_INTERVAL} batches (20% of epoch)\n")

global_batch          = 0
best_epoch_loss       = float("inf")
best_step_loss        = float("inf")
epochs_no_improve     = 0
checkpoint_loss_accum = 0.0
checkpoint_batch_count = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"{'='*60}")
    print(f"  EPOCH {epoch}/{NUM_EPOCHS}")
    print(f"{'='*60}")

    model.train()
    epoch_loss   = 0.0
    batch_count  = 0
    accum_step   = 0
    batch_buffer = []

    stream       = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=epoch)
    progress_bar = tqdm(stream, desc=f"Epoch {epoch}", total=TOTAL_SAMPLES, leave=True)

    for sample in progress_bar:
        batch_buffer.append(sample)

        if len(batch_buffer) < BATCH_SIZE:
            continue

        paragraph_embs, text_input_ids, attention_mask = prepare_batch(batch_buffer)
        paragraph_embs  = paragraph_embs.to(device)
        text_input_ids  = text_input_ids.to(device)
        attention_mask  = attention_mask.to(device)
        batch_buffer    = []

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lm_loss, aux_loss = model(paragraph_embs, text_input_ids, attention_mask)
            loss = (lm_loss + AUX_LOSS_WEIGHT * aux_loss) / GRAD_ACCUM

        loss.backward()
        accum_step += 1

        if accum_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        raw_loss      = (lm_loss.item() + AUX_LOSS_WEIGHT * aux_loss.item())
        epoch_loss   += raw_loss
        batch_count  += 1
        global_batch += 1

        checkpoint_loss_accum  += raw_loss
        checkpoint_batch_count += 1

        wandb.log({
            "train_loss": raw_loss,
            "lm_loss":    lm_loss.item(),
            "aux_loss":   aux_loss.item(),
            "lr_mlp":     scheduler.get_last_lr()[0],
            "lr_lora":    scheduler.get_last_lr()[1],
            "epoch":      epoch,
            "batch":      global_batch,
        })

        progress_bar.set_postfix({"loss": f"{raw_loss:.4f}"})

        # Terminal log at fixed interval
        if batch_count % LOG_INTERVAL == 0:
            avg_so_far = epoch_loss / batch_count
            print(
                f"  [Epoch {epoch} | Batch {batch_count:>4}/{BATCHES_PER_EPOCH}]  "
                f"loss: {raw_loss:.4f}  avg: {avg_so_far:.4f}  "
                f"lr_mlp: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Mid-epoch checkpoint — compare an AVERAGE over the checkpoint window,
        # not a single noisy batch loss, since a single batch is not a reliable
        # signal of "best" performance.
        if batch_count % CHECKPOINT_INTERVAL == 0:
            pct = (batch_count / BATCHES_PER_EPOCH) * 100
            avg_checkpoint_loss = checkpoint_loss_accum / max(checkpoint_batch_count, 1)
            checkpoint_loss_accum  = 0.0
            checkpoint_batch_count = 0

            if avg_checkpoint_loss < best_step_loss - ES_MIN_DELTA:
                best_step_loss = avg_checkpoint_loss
                torch.save(model.state_dict(), BEST_STEPS_PATH)

                print(
                    f"\n  [Best Step Checkpoint @ {pct:.0f}%] "
                    f"window avg improved to {avg_checkpoint_loss:.4f} → saved to best_steps_checkpoint.pt"
                )
            else:
                print(
                    f"\n  [Step Checkpoint @ {pct:.0f}%] "
                    f"no improvement (window avg: {avg_checkpoint_loss:.4f})"
                )

        if batch_count >= BATCHES_PER_EPOCH:
            break

    avg_epoch_loss = epoch_loss / max(batch_count, 1)
    print(f"\n  Epoch {epoch} complete — avg loss: {avg_epoch_loss:.4f}")
    wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch})

    # Save best model and check early stopping
    if avg_epoch_loss < best_epoch_loss - ES_MIN_DELTA:
        best_epoch_loss   = avg_epoch_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_EPOCH_PATH)
        print(f"  New best loss {best_epoch_loss:.4f} — checkpoint saved.")
    else:
        epochs_no_improve += 1
        print(f"  No improvement ({epochs_no_improve}/{ES_PATIENCE}).")

    if epochs_no_improve >= ES_PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch}.")
        break

print("\nTraining complete.")
wandb.finish()

# ==========================================
# UPLOAD BEST CHECKPOINT TO HUGGINGFACE
# ==========================================
api = HfApi()
api.create_repo(repo_id=TARGET_REPO, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj=BEST_EPOCH_PATH,
    path_in_repo="best_epoch_checkpoint.pt",
    repo_id=TARGET_REPO,
    repo_type="model",
)
api.upload_file(
    path_or_fileobj=BEST_STEPS_PATH,
    path_in_repo="best_steps_checkpoint.pt",
    repo_id=TARGET_REPO,
    repo_type="model",
)
print("Upload complete.")