import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
import os

CHUNKS = "data/processed/chunks.csv"
OUTPUT = "data/indexes/bm25_index.pkl"

df = pd.read_csv(CHUNKS)

# tokenize each chunk
tokenized_corpus = [
    nltk.word_tokenize(text.lower())
    for text in df["chunk_text"]
]

bm25 = BM25Okapi(tokenized_corpus)
os.makedirs("data/indexes", exist_ok=True)

with open(OUTPUT, "wb") as f:
    pickle.dump(bm25, f)

print(f"Indexed {len(df)} chunks.")