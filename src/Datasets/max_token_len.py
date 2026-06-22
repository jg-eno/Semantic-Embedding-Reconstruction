import h5py

FILE = "corrected_msmarco_token_sentence_embeddings.h5"

with h5py.File(FILE) as f:
    print(f.keys())
    print(sum(f["seq_lengths"][:])/len(f["seq_lengths"][:]))

#Output : 252