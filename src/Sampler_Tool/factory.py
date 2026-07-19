import gc
import re

import torch
from huggingface_hub import hf_hub_download
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from config import DECODER_BASE_NAME, DECODER_LORA_DROPOUT, DEVICE
from models import EndToEndInverter

_LORA_KEY_PATTERN = re.compile(r"\.([a-zA-Z0-9_]+)\.lora_(A|B)\.default\.weight$")
_MLP_HEAD_PATTERN = re.compile(r"^m_parallel_mlps\.(\d+)\.net\.0\.weight$")


def detect_lora_config(state_dict: dict) -> dict:
    """Infer target_modules and rank r from checkpoint keys."""
    target_modules: set[str] = set()
    ranks_seen: set[int] = set()

    for key, tensor in state_dict.items():
        match = _LORA_KEY_PATTERN.search(key)
        if match is None:
            continue
        proj_name, ab = match.group(1), match.group(2)
        target_modules.add(proj_name)
        ranks_seen.add(tensor.shape[0] if ab == "A" else tensor.shape[1])

    if not target_modules:
        raise ValueError("No LoRA weights found in checkpoint.")
    if len(ranks_seen) > 1:
        raise ValueError(f"Multiple LoRA ranks in checkpoint: {sorted(ranks_seen)}")

    return {"target_modules": sorted(target_modules), "r": ranks_seen.pop()}


def detect_mlp_config(state_dict: dict) -> dict:
    """Infer prefix_len, mlp_hidden_dim, and MLP weight dtype from checkpoint keys."""
    max_idx = -1
    hidden_dim = 2048
    dtype = torch.float32

    for key, tensor in state_dict.items():
        m = _MLP_HEAD_PATTERN.match(key)
        if m:
            idx = int(m.group(1))
            max_idx = max(max_idx, idx)
            hidden_dim = tensor.shape[0]   # net.0.weight: [hidden, in]
            dtype = tensor.dtype

    if max_idx < 0:
        raise ValueError("No MLP heads found in checkpoint.")

    return {"prefix_len": max_idx + 1, "mlp_hidden_dim": hidden_dim, "mlp_dtype": dtype}


class InverterFactory:
    """Downloads a checkpoint and assembles a loaded EndToEndInverter."""

    def __init__(self, base_name: str = DECODER_BASE_NAME):
        self.base_name = base_name

    def _build_decoder(self, target_modules: list[str], r: int, lora_alpha: int | None = None):
        # fp16 halves the decoder footprint (~1.1 GB instead of ~2.2 GB).
        base = AutoModelForCausalLM.from_pretrained(
            self.base_name,
            dtype=torch.float16,
            device_map={"": DEVICE},
            trust_remote_code=True,
        )
        alpha = lora_alpha if lora_alpha is not None else 2 * r
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=DECODER_LORA_DROPOUT,
            target_modules=target_modules,
        )
        print(f"  [Factory] target_modules={target_modules}  r={r}  alpha={alpha}")
        return get_peft_model(base, lora_cfg)

    def load(
        self,
        repo: str,
        filename: str,
        paragraph_dim: int,
        # prefix_len is now auto-detected from the checkpoint; kept as an
        # optional override for backwards-compat with callers that pass it.
        prefix_len: int | None = None,
        lora_alpha: int | None = None,
    ) -> EndToEndInverter:
        """Download checkpoint, auto-detect all architecture params, return eval-mode model."""
        print(f"  [Factory] downloading {filename} from {repo} …")
        state_dict = torch.load(
            hf_hub_download(repo_id=repo, filename=filename),
            map_location="cpu",
        )

        mlp_cfg  = detect_mlp_config(state_dict)
        lora_cfg = detect_lora_config(state_dict)

        effective_prefix_len = prefix_len or mlp_cfg["prefix_len"]
        print(
            f"  [Factory] prefix_len={effective_prefix_len}  "
            f"mlp_hidden_dim={mlp_cfg['mlp_hidden_dim']}  "
            f"mlp_dtype={mlp_cfg['mlp_dtype']}"
        )

        # Decoder loads directly onto GPU in fp16.
        decoder = self._build_decoder(
            target_modules=lora_cfg["target_modules"],
            r=lora_cfg["r"],
            lora_alpha=lora_alpha,
        )

        # MLPs built on CPU first, weights loaded, then moved to GPU in their
        # native dtype (bf16 in v4) to minimise VRAM and avoid dtype mismatches.
        model = EndToEndInverter(
            paragraph_dim=paragraph_dim,
            decoder_hidden_dim=decoder.config.hidden_size,
            prefix_len=effective_prefix_len,
            decoder_model=decoder,
            mlp_hidden_dim=mlp_cfg["mlp_hidden_dim"],
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  [Factory] missing={len(missing)}  unexpected={len(unexpected)}")
        if missing:
            print("    missing:", missing[:3])
        if unexpected:
            print("    unexpected:", unexpected[:3])

        # Move only the MLP heads; decoder is already on GPU via device_map.
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        model.m_parallel_mlps.to(device=DEVICE, dtype=mlp_cfg["mlp_dtype"])

        return model.eval()
