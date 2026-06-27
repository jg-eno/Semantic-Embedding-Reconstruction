"""
Architecture sweep for the prefix MLPs.

Searches over:
  - prefix_len          (no. of prefix tokens fed to the decoder)
  - num_hidden_layers   (depth of each SingleTokenMLP)
  - hidden_dim          (width of each hidden layer, shared across layers)

Run on a 20k-sample subset for 5 epochs per trial — this is a PROXY search,
not a substitute for the full run. Pick the winning config here, then
re-train at full scale (100k samples/epoch, GPU_Scripts/gpu_training_script.py)
using the winning (prefix_len, num_hidden_layers, hidden_dim).

Usage:
    wandb sweep sweep_config.yaml      # prints a sweep ID
    wandb agent <sweep_id>             # runs trials, one per process
"""

import os
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
from huggingface_hub import login
from dotenv import load_dotenv

# ==========================================
# FIXED ACROSS ALL TRIALS (not swept)
# ==========================================
MODEL_NAME      = "Qwen/Qwen3-0.6B"
PARAGRAPH_DIM   = 1024
MAX_TEXT_LEN    = 128
BATCH_SIZE      = 64
GRAD_ACCUM      = 2
LEARNING_RATE   = 3e-4
TOTAL_SAMPLES   = 20_000     # proxy subset, not the full 100k-per-epoch production run
NUM_EPOCHS      = 4
DATASET         = "jg-eno/MSMACRO-1M-Qwen-Embeddings"
AUX_LOSS_WEIGHT = 1
LOG_INTERVAL    = 20

BATCHES_PER_EPOCH = TOTAL_SAMPLES // BATCH_SIZE

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# PARAMETERIZED PREFIX MLP
# (this is the only architectural change vs. the production script —
#  hidden_dims is now a tuple instead of a hardcoded single 2048 layer)
# ==========================================
class SingleTokenMLP(nn.Module):
    def __init__(self, paragraph_dim: int, decoder_hidden_dim: int,
                 hidden_dims: tuple, dropout: float = 0.1):
        super().__init__()
        dims = [paragraph_dim, *hidden_dims, decoder_hidden_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # no activation/dropout after the final linear
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EndToEndInverter(nn.Module):
    def __init__(self, paragraph_dim, decoder_hidden_dim, prefix_len,
                 hidden_dims, decoder_model):
        super().__init__()
        self.prefix_len = prefix_len
        self.m_parallel_mlps = nn.ModuleList([
            SingleTokenMLP(paragraph_dim, decoder_hidden_dim, hidden_dims)
            for _ in range(prefix_len)
        ])
        self.decoder = decoder_model

    def forward(self, paragraph_embs, text_input_ids, attention_mask):
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


def prepare_batch(batch_samples, tokenizer):
    paragraph_embs_list, input_ids_list, attention_mask_list = [], [], []
    for sample in batch_samples:
        sent_emb = torch.tensor(sample["sentence_embeddings"], dtype=torch.float32)
        paragraph_embs_list.append(sent_emb)

        ids     = sample["input_ids"]
        seq_len = min(len(ids), MAX_TEXT_LEN)

        padded_ids = np.full(MAX_TEXT_LEN, tokenizer.pad_token_id, dtype=np.int64)
        padded_ids[:seq_len] = ids[:seq_len]
        input_ids_list.append(torch.tensor(padded_ids))

        mask = np.zeros(MAX_TEXT_LEN, dtype=np.float32)
        mask[:seq_len] = 1.0
        attention_mask_list.append(torch.tensor(mask))

    return (
        torch.stack(paragraph_embs_list),
        torch.stack(input_ids_list),
        torch.stack(attention_mask_list),
    )


def run_trial():
    wandb.init()
    cfg = wandb.config

    prefix_len        = cfg.prefix_len
    num_hidden_layers  = cfg.num_hidden_layers
    hidden_dim         = cfg.hidden_dim
    hidden_dims        = tuple([hidden_dim] * num_hidden_layers)

    print(f"\nTrial config: prefix_len={prefix_len}, "
          f"num_hidden_layers={num_hidden_layers}, hidden_dim={hidden_dim}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_decoder = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device},
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    decoder_with_lora = get_peft_model(base_decoder, lora_config)

    model = EndToEndInverter(
        paragraph_dim=PARAGRAPH_DIM,
        decoder_hidden_dim=base_decoder.config.hidden_size,
        prefix_len=prefix_len,
        hidden_dims=hidden_dims,
        decoder_model=decoder_with_lora,
    )
    model.m_parallel_mlps.to(device=device, dtype=torch.bfloat16)

    # Log MLP parameter count — useful for reading the sweep results table
    # alongside loss (a bigger MLP "winning" by a tiny margin may not be worth it)
    mlp_param_count = sum(p.numel() for p in model.m_parallel_mlps.parameters())
    wandb.log({"mlp_param_count": mlp_param_count})

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

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(BATCHES_PER_EPOCH * NUM_EPOCHS) // GRAD_ACCUM,
        eta_min=LEARNING_RATE * 0.05,
    )

    best_epoch_loss = float("inf")
    global_batch    = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, batch_count, accum_step = 0.0, 0, 0
        batch_buffer = []

        stream = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=epoch)
        progress_bar = tqdm(stream, desc=f"Epoch {epoch}", total=TOTAL_SAMPLES, leave=False)

        for sample in progress_bar:
            batch_buffer.append(sample)
            if len(batch_buffer) < BATCH_SIZE:
                continue

            paragraph_embs, text_input_ids, attention_mask = prepare_batch(batch_buffer, tokenizer)
            paragraph_embs = paragraph_embs.to(device)
            text_input_ids = text_input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_buffer   = []

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

            raw_loss = lm_loss.item() + AUX_LOSS_WEIGHT * aux_loss.item()
            epoch_loss   += raw_loss
            batch_count  += 1
            global_batch += 1

            wandb.log({
                "train_loss": raw_loss,
                "lm_loss":    lm_loss.item(),
                "aux_loss":   aux_loss.item(),
                "epoch":      epoch,
                "batch":      global_batch,
            })
            progress_bar.set_postfix({"loss": f"{raw_loss:.4f}"})

            if batch_count % LOG_INTERVAL == 0:
                print(f"  [Epoch {epoch} | Batch {batch_count}/{BATCHES_PER_EPOCH}] "
                      f"loss: {raw_loss:.4f}  avg: {epoch_loss / batch_count:.4f}")

            if batch_count >= BATCHES_PER_EPOCH:
                break

        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        best_epoch_loss = min(best_epoch_loss, avg_epoch_loss)
        print(f"  Epoch {epoch} avg loss: {avg_epoch_loss:.4f}")
        wandb.log({"epoch_avg_loss": avg_epoch_loss, "epoch": epoch})

    # Final metric the sweep optimizes against
    wandb.log({"best_epoch_loss": best_epoch_loss})
    wandb.finish()


if __name__ == "__main__":
    run_trial()
