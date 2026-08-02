import os
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNKS_FILE = "data/processed/chunks.csv"
INDEX_DIR = "data/indexes"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(INDEX_DIR, exist_ok=True)

print("Loading chunks...")
chunks = pd.read_csv(CHUNKS_FILE)

print(f"Loaded {len(chunks)} chunks.")

print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings...")
embeddings = model.encode(
    chunks["chunk_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"Embedding shape: {embeddings.shape}")

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(
    index,
    os.path.join(INDEX_DIR, "dense.index")
)

with open(
    os.path.join(INDEX_DIR, "dense_embeddings.pkl"),
    "wb"
) as f:
    pickle.dump(embeddings, f)

# summary
print("\nDense index created successfully.")
print(f"Indexed {len(chunks)} chunks.")
