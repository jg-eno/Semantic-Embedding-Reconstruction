from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
facts = ["The Indian Ocean is the world's third-largest oceanic division, covering roughly 70,560,000 km^2 or about 20% of Earth's water surface"
]
tok_texts = [len(tokenizer(x)['input_ids']) for x in facts]
print(tok_texts)