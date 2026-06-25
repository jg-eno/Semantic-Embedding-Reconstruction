import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
import re
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="transformers"
)

CONFIG = {
        "repo": "jg-eno/ReLoDer_v3",
        "prefix_len": 128,
        "filename": "best_steps_checkpoint.pt"
    }

SAMPLE_PARAGRAPH = "The Sun rises in the east and sets in the west"
MAX_NEW_TOKENS = 20


device = "cuda" if torch.cuda.is_available() else "cpu"

class SingleTokenMLP(nn.Module):
    def __init__(self, paragraph_dim, decoder_hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(paragraph_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, decoder_hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class EndToEndInverter(nn.Module):
    def __init__(self, paragraph_dim, decoder_hidden_dim, prefix_len, decoder_model):
        super().__init__()
        self.prefix_len = prefix_len
        self.m_parallel_mlps = nn.ModuleList(
            [SingleTokenMLP(paragraph_dim, decoder_hidden_dim) for _ in range(prefix_len)]
        )
        self.decoder = decoder_model

    def _prefix(self, paragraph_embs):
        return torch.stack([mlp(paragraph_embs) for mlp in self.m_parallel_mlps], dim=1)

    @torch.no_grad()
    def generate(self, paragraph_embs, tokenizer, max_new_tokens=128):
        paragraph_embs = paragraph_embs.to(next(self.parameters()).device)
        prefix = self._prefix(paragraph_embs).to(dtype=self.decoder.dtype)
        outputs = self.decoder.generate(
            inputs_embeds=prefix,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.2,
        )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Matches keys like:
#   decoder.base_model.model.model.layers.19.self_attn.q_proj.lora_A.default.weight
LORA_KEY_PATTERN = re.compile(r"\.([a-zA-Z0-9_]+)\.lora_(A|B)\.default\.weight$")


def detect_lora_config(state_dict):
    """
    Recovers target_modules and rank r directly from a checkpoint's state dict,
    instead of assuming a single hardcoded LoraConfig for every checkpoint.
    """

    target_modules = set()
    ranks_seen = set()

    for key, tensor in state_dict.items():
        match = LORA_KEY_PATTERN.search(key)
        if match is None:
            continue
        proj_name, ab = match.group(1), match.group(2)
        target_modules.add(proj_name)
        # lora_A: [r, in_dim] -> r is dim 0 | lora_B: [out_dim, r] -> r is dim 1
        ranks_seen.add(tensor.shape[0] if ab == "A" else tensor.shape[1])

    if not target_modules:
        raise ValueError(
            "No LoRA weights found in this checkpoint (no '*.lora_A/B.default.weight' "
            "keys). Cannot auto-detect a LoRA config for it."
        )
    if len(ranks_seen) > 1:
        raise ValueError(f"Multiple LoRA ranks found in one checkpoint: {sorted(ranks_seen)}")

    return {"target_modules": sorted(target_modules), "r": ranks_seen.pop()}


class InverterFactory:
    def __init__(self):
        self.base_name = "Qwen/Qwen3-0.6B"

    def build_decoder(self, target_modules, r, lora_alpha=None):
        base = AutoModelForCausalLM.from_pretrained(
            self.base_name, device_map={"": 0}, trust_remote_code=True
        )

        alpha = lora_alpha if lora_alpha is not None else 2 * r

        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
        )

        print(f"  Building decoder with target_modules={target_modules}, r={r}, alpha={alpha}")
        return get_peft_model(base, lora)

    def load(self, repo, filename, prefix_len, paragraph_dim, lora_alpha=None):
        # Download and inspect the checkpoint FIRST, so the decoder we build
        # actually matches what's inside it, instead of guessing up front.
        state_dict = torch.load(
            hf_hub_download(repo_id=repo, filename=filename),
            map_location="cpu",
        )

        detected = detect_lora_config(state_dict)
        decoder = self.build_decoder(
            target_modules=detected["target_modules"],
            r=detected["r"],
            lora_alpha=lora_alpha,
        )

        model = EndToEndInverter(
            paragraph_dim=paragraph_dim,
            decoder_hidden_dim=decoder.config.hidden_size,
            prefix_len=prefix_len,
            decoder_model=decoder,
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(repo, "missing:", len(missing), "unexpected:", len(unexpected))
        if missing or unexpected:
            print("  missing keys sample:", missing[:3])
            print("  unexpected keys sample:", unexpected[:3])

        return model.to(device).eval()
    
class Encoder:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        return F.normalize(sentence_embeddings, p=2, dim=1)
    
class Runner:
    def __init__(self):
        self.encoder = Encoder()
        self.factory = InverterFactory()

    def run(self, cfg, texts, max_new_tokens):
        embeddings = self.encoder.encode(texts)
        tokenizer = self.encoder.tokenizer

        model = self.factory.load(
                repo=cfg["repo"],
                filename=cfg["filename"],
                prefix_len=cfg["prefix_len"],
                paragraph_dim=embeddings.shape[1],
            )

        outputs = model.generate(embeddings, tokenizer, max_new_tokens=max_new_tokens)

        #outputs = [re.sub(r'[^\x00-\x7F]+', '', text) for text in outputs]
        print("Original Sentence : ",SAMPLE_PARAGRAPH)
        print("Model Output : ",outputs[0])

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    runner = Runner()
    runner.run(CONFIG, SAMPLE_PARAGRAPH, MAX_NEW_TOKENS)