import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import DEVICE, ENCODER_MODEL_NAME, ENCODER_MAX_LENGTH  # DEVICE used for initial load


class Encoder:
    """Mean-pooled, L2-normalised sentence encoder."""

    def __init__(self, model_name: str = ENCODER_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    @torch.no_grad()
    def encode(self, texts: list[str] | str) -> torch.Tensor:
        """Returns a (batch, hidden_dim) normalised embedding tensor."""
        if isinstance(texts, str):
            texts = [texts]

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ENCODER_MAX_LENGTH,
        ).to(device)

        outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        sentence_embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)

        return F.normalize(sentence_embeddings, p=2, dim=1)
