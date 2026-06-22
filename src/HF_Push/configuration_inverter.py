from transformers import PretrainedConfig

class EmbeddingInverterConfig(PretrainedConfig):
    model_type = "embedding_inverter"

    def __init__(
        self,
        paragraph_dim=1024,
        decoder_hidden_dim=1024,   
        prefix_len=32,
        decoder_model_name="Qwen/Qwen3-0.6B",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.paragraph_dim = paragraph_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.prefix_len = prefix_len
        self.decoder_model_name = decoder_model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "v_proj"]