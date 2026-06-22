import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from configuration_inverter import EmbeddingInverterConfig


class SingleTokenMLP(nn.Module):
    def __init__(self, paragraph_dim, decoder_hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(paragraph_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, decoder_hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingInverterModel(PreTrainedModel):
    config_class = EmbeddingInverterConfig
    _tied_weights_keys = {}

    def __init__(self, config: EmbeddingInverterConfig):
        super().__init__(config)

        self.prefix_mlps = nn.ModuleList([
            SingleTokenMLP(config.paragraph_dim, config.decoder_hidden_dim)
            for _ in range(config.prefix_len)
        ])

        self._decoder = None
        self._tokenizer = None

    def _load_decoder(self):
        """Lazily load decoder + LoRA. Safe to call multiple times."""
        if self._decoder is not None:
            return

        config = self.config
        base = AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name,
            trust_remote_code=True,
        )
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        self._decoder = get_peft_model(base, lora).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_name)

    def _prefix(self, paragraph_embs):
        return torch.stack([mlp(paragraph_embs) for mlp in self.prefix_mlps], dim=1)

    def forward(self, paragraph_embs):
        return self._prefix(paragraph_embs)

    @torch.no_grad()
    def invert(self, paragraph_embs: torch.Tensor, max_new_tokens: int = 128):
        self._load_decoder()   # materializes decoder only here
        self.eval()

        paragraph_embs = paragraph_embs.to(self.device)
        prefix = self._prefix(paragraph_embs).to(dtype=self._decoder.dtype)

        outputs = self._decoder.generate(
            inputs_embeds=prefix,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.2,
        )
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)