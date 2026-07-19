import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "repo": "Subhav-K/ReLoDer_v4",
    "prefix_len": 64,
    "filename": "best_steps_checkpoint.pt",
}

ENCODER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
ENCODER_MAX_LENGTH = 512

DECODER_BASE_NAME = "Qwen/Qwen3-0.6B"
DECODER_LORA_DROPOUT = 0.05

MAX_NEW_TOKENS = 20

SAMPLER_DEFAULTS = {
    "temperature": 1.4,       # raised further per-sample for diversity
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.3,
    "temperature_step": 0.05, # added to temperature each successive sample
}

SAMPLE_PARAGRAPH = "The Sun rises in the east and sets in the west"
