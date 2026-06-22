import torch
from huggingface_hub import hf_hub_download

from configuration_inverter import EmbeddingInverterConfig
from modeling_inverter import EmbeddingInverterModel

config = EmbeddingInverterConfig(
    paragraph_dim=1024,
    decoder_hidden_dim=1024,
    prefix_len=32,
    decoder_model_name="Qwen/Qwen3-0.6B",
)

EmbeddingInverterConfig.register_for_auto_class()
EmbeddingInverterModel.register_for_auto_class("AutoModel")

model = EmbeddingInverterModel(config)

old_state = torch.load(
    hf_hub_download(
        repo_id="Subhav-K/qwen-embedding-inverter-v2",
        filename="new_end_to_end_aux_model.pt",
    ),
    map_location="cpu",
)

remapped = {
    k.replace("m_parallel_mlps.", "prefix_mlps.").replace("decoder.", "decoder."): v
    for k, v in old_state.items()
}

missing, unexpected = model.load_state_dict(remapped, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

REPO = "jg-eno/Qwen-Embedding-Inverter"
model.push_to_hub(REPO)
config.push_to_hub(REPO)

from huggingface_hub import HfApi
api = HfApi()
api.upload_file(path_or_fileobj="configuration_inverter.py", path_in_repo="configuration_inverter.py", repo_id=REPO)
api.upload_file(path_or_fileobj="modeling_inverter.py", path_in_repo="modeling_inverter.py", repo_id=REPO)
