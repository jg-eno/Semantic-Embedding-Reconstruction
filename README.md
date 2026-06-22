# Cuda Driver Installation
sudo apt update
sudo apt install nvidia-driver-535

# Experiment-1: Semantic Embedding Expander

## Goal
Map a sentence embedding `[D]` → token embeddings `[N, D]`  
so that decoded text is **semantically similar** to the original.

---

## Model

### Sentence Encoder
- Input: `[B, D]`
- Output: `[B, hidden_dim]`
- MLP with residual blocks

### Positional Expander
- Input: `[B, hidden_dim]`
- Output: `[B, N, D]`
- Broadcast context to all positions
- Add learned positional embeddings
- Decode each position independently

### Length Head
- Predicts normalized sequence length

---

## Loss

Combined loss:

- Token cosine → match token embeddings  
- Sentence cosine → match overall meaning  
- Smoothness → reduce noise between tokens  
- Length → predict sequence length  

```

L = λ₁·token + λ₂·sentence + λ₃·smooth + λ₄·length

```

---

## Decoding

Token embeddings → nearest tokens using vocab embeddings

### Methods:
- **Greedy**: argmax per position (fast, repetitive)
- **Sampling**: top-k + temperature (more diverse, noisy)
- **Beam**: best quality, slow

---

## Evaluation

1. Generate token embeddings  
2. Decode to text  
3. Re-encode text  
4. Compute cosine similarity with original embedding  

---

## Results (Beam, 200 samples)

- Mean cosine: **0.82**
- > 0.80: **71.5%**
- > 0.70: **98.5%**

---

## Observations

- Captures **keywords and topic**
- Fails at:
  - grammar
  - ordering
  - repetition control

---

## Why it failed

- Tokens generated **independently**
- No sequence modeling (no token interaction)
- Input embedding loses word order
- Objective optimizes **similarity**, not **language**
- Nearest-neighbor decoding adds noise

---

## Takeaway

- Works for **semantic reconstruction**
- Fails for **text generation**


#To try out :
- Different Loss Function
- K : 32 -> 64
- Increase the dataset size : 10k -> 1M (1.1 -> 2.1)