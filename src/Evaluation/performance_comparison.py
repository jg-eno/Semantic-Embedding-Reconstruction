import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, LoraConfig, TaskType


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


class InverterFactory:
    def __init__(self):
        self.base_name = "Qwen/Qwen3-0.6B"

    def build_decoder(self):
        base = AutoModelForCausalLM.from_pretrained(
            self.base_name, device_map={"": 0}, trust_remote_code=True
        )

        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )

        return get_peft_model(base, lora)

    def load(self, repo, filename, prefix_len, paragraph_dim):
        decoder = self.build_decoder()

        model = EndToEndInverter(
            paragraph_dim=paragraph_dim,
            decoder_hidden_dim=decoder.config.hidden_size,
            prefix_len=prefix_len,
            decoder_model=decoder,
        )

        state_dict = torch.load(
            hf_hub_download(repo_id=repo, filename=filename),
            map_location="cpu",
        )

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(repo, len(missing), len(unexpected))

        return model.to(device).eval()


class Evaluator:
    def __init__(self):
        self.sim = SentenceTransformer("all-MiniLM-L6-v2").to(device)

    def score(self, inputs, outputs):
        scores = []
        for i in range(len(inputs)):
            a = self.sim.encode(inputs[i], convert_to_tensor=True).to(device)
            b = self.sim.encode(outputs[i], convert_to_tensor=True).to(device)
            scores.append(F.cosine_similarity(a, b, dim=0).item())
        return scores


class Runner:
    def __init__(self):
        self.encoder = Encoder()
        self.factory = InverterFactory()
        self.evaluator = Evaluator()

    def run(self, configs, texts):
        embeddings = self.encoder.encode(texts)
        tokenizer = self.encoder.tokenizer

        results = {}

        for cfg in configs:
            model = self.factory.load(
                repo=cfg["repo"],
                filename=cfg["filename"],
                prefix_len=cfg["prefix_len"],
                paragraph_dim=embeddings.shape[1],
            )

            outputs = model.generate(embeddings, tokenizer, max_new_tokens=40)
            scores = self.evaluator.score(texts, outputs)

            results[cfg["repo"]] = {
                "outputs": outputs,
                "scores": scores,
                "avg": sum(scores) / len(scores),
            }

            del model
            torch.cuda.empty_cache()

        return results
    
    def report(self, texts, results):
        names = list(results.keys())

        for i in range(len(texts)):
            print(f"\n=== Sample {i+1} ===")
            print("Input:", texts[i])
            for n in names:
                print(n, results[n]["outputs"][i])
                print("Cos:", round(results[n]["scores"][i], 4))

        print("\n=== Averages ===")
        for n in names:
            print(n, results[n]["avg"])


facts = [
    "Paris is the capital of France and it is known for its rich cultural heritage, world-renowned art museums, historic architecture, and famous landmarks like the Eiffel Tower and the Louvre Museum, which houses thousands of works including the Mona Lisa, making the city one of the most visited tourist destinations in the world and a global center for fashion, cuisine, and intellectual life, attracting millions of visitors every year who come to experience its vibrant street life, historical neighborhoods, iconic cafes, and its lasting influence on art, literature, and global culture throughout history",

    "The Earth revolves around the Sun once every year in an elliptical orbit, and this motion, along with its tilted axis, is responsible for the changing seasons we experience, influencing variations in temperature, daylight hours, and weather patterns across different regions of the planet throughout the year, while also playing a crucial role in sustaining life by regulating climate systems, supporting ecosystems, and enabling the natural cycles that govern agriculture, biodiversity, and environmental balance",

    "Water boils at one hundred degrees Celsius under standard atmospheric pressure, and this physical property is widely used in cooking, scientific experiments, and industrial processes, while also serving as a reference point in temperature measurement systems and changing slightly with variations in altitude and pressure, making it an essential concept in thermodynamics and practical applications where precise temperature control and understanding of phase changes are critical for efficiency and safety",

    "The human brain controls the entire body by sending electrical and chemical signals through the nervous system, allowing us to think, move, feel emotions, process sensory information, store memories, make decisions, and respond effectively to our environment in both conscious and unconscious ways, while also coordinating complex bodily functions such as breathing, heartbeat, and hormonal regulation, making it one of the most sophisticated and vital organs in the human body",

    "Mount Everest is the highest mountain above sea level, standing at over 8800 meters, and it is located in the Himalayas on the border between Nepal and China, attracting climbers from around the world despite its extreme conditions, including low oxygen levels, freezing temperatures, and challenging terrain, and it remains a symbol of human endurance and exploration as expeditions continue to push the limits of physical and mental capability in one of the harshest environments on Earth"
]

configs = [
    {
        "repo": "Subhav-K/qwen-embedding-inverter-v7",
        "prefix_len": 32,
        "filename": "new_end_to_end_model.pt"
    },
    {
        "repo": "Subhav-K/qwen-embedding-inverter-v2",
        "prefix_len": 32,
        "filename": "new_end_to_end_aux_model.pt"
    },
]

runner = Runner()
results = runner.run(configs, facts)
runner.report(facts, results)