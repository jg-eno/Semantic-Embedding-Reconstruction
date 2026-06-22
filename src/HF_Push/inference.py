import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DECODER_MODEL_NAME = "Qwen/Qwen3-0.6B"
INVERTER_MODEL_NAME = "jg-eno/Qwen-Embedding-Inverter"

USE_8BIT = False   # set True if still OOM

# =========================
# UTIL: CLEAN GPU
# =========================
def clean_gpu():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# =========================
# STEP 1: ENCODER
# =========================
class Encoder:
    def __init__(self, model_name):
        self.device = DEVICE
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=DTYPE
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()

        sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

        print("Embedding shape:", sentence_embeddings.shape)

        return F.normalize(sentence_embeddings, p=2, dim=1)

# =========================
# INPUT
# =========================
sample_text = (
    "The Indian Ocean is the world's third-largest oceanic division, "
    "covering roughly 70,560,000 km^2 or about 20% of Earth's water surface"
)

# =========================
# RUN ENCODER
# =========================
print("\n--- Encoding ---")
encoder = Encoder(EMBED_MODEL_NAME)
embedding = encoder.encode(sample_text)

# Move embedding OFF GPU before loading next model
embedding = embedding.cpu()

del encoder
clean_gpu()

# =========================
# STEP 2: INVERTER BASELINE
# =========================
print("\n--- Inverter Baseline ---")

inverter = AutoModel.from_pretrained(
    INVERTER_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE
).to(DEVICE).eval()

with torch.no_grad():
    inv_result = inverter.invert(
        embedding.to(DEVICE).to(DTYPE),
        max_new_tokens=40
    )[0]

print("Inverter Output:\n", inv_result)

# cleanup
del inverter
clean_gpu()

# =========================
# STEP 3: DIRECT DECODER BASELINE
# =========================
print("\n--- Direct Decoder Baseline ---")

# Optional 8-bit config
quant_config = None
if USE_8BIT:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

lm = AutoModelForCausalLM.from_pretrained(
    DECODER_MODEL_NAME,
    torch_dtype=DTYPE if not USE_8BIT else None,
    quantization_config=quant_config,
    device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_NAME)

emb = embedding.to(DEVICE).to(DTYPE)

hidden_size = lm.config.hidden_size
emb_dim = emb.shape[-1]

if emb_dim != hidden_size:
    print(f"Projection needed: {emb_dim} -> {hidden_size}")
    projector = torch.nn.Linear(emb_dim, hidden_size).to(DEVICE).to(DTYPE)
    emb = projector(emb)

# Try multiple variants
def generate_from_embedding(emb, repeat=1):
    inputs_embeds = emb.unsqueeze(1).repeat(1, repeat, 1)
    inputs_embeds = inputs_embeds.to(lm.dtype)
    attention_mask = torch.ones(inputs_embeds.shape[:2]).to(DEVICE)

    with torch.no_grad():
        outputs = lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=40,
            use_cache=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Variant 1: single token
out1 = generate_from_embedding(emb, repeat=1)

# Variant 2: repeated embedding
out2 = generate_from_embedding(emb, repeat=10)

print("\n[Single Token Output]\n", out1)
print("\n[Repeated Embedding Output]\n", out2)

# cleanup
del lm
clean_gpu()

print("\n--- Done ---")