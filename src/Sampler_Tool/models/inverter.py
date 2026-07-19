import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from models.mlp import SingleTokenMLP


class EndToEndInverter(nn.Module):
    """
    Maps a paragraph embedding to text via a soft prefix fed into a LoRA decoder.

    paragraph_emb → [MLP_0 … MLP_{prefix_len-1}] → prefix → decoder → tokens
    """

    def __init__(
        self,
        paragraph_dim: int,
        decoder_hidden_dim: int,
        prefix_len: int,
        decoder_model: nn.Module,
        mlp_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.m_parallel_mlps = nn.ModuleList(
            [SingleTokenMLP(paragraph_dim, decoder_hidden_dim, mlp_hidden_dim) for _ in range(prefix_len)]
        )
        self.decoder = decoder_model

    def _prefix(self, paragraph_embs: torch.Tensor) -> torch.Tensor:
        """Returns (batch, prefix_len, decoder_hidden_dim) on the decoder's device in its dtype."""
        raw = torch.stack(
            [mlp(paragraph_embs) for mlp in self.m_parallel_mlps], dim=1
        )
        decoder_device = next(self.decoder.parameters()).device
        return raw.to(device=decoder_device, dtype=self.decoder.dtype)

    @torch.no_grad()
    def generate(
        self,
        paragraph_embs: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.2,
    ) -> list[str]:
        mlp_device = next(self.m_parallel_mlps.parameters()).device
        mlp_dtype  = next(self.m_parallel_mlps.parameters()).dtype
        paragraph_embs = paragraph_embs.to(device=mlp_device, dtype=mlp_dtype)
        prefix = self._prefix(paragraph_embs)

        outputs = self.decoder.generate(
            inputs_embeds=prefix,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
