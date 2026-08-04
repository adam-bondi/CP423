import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

TOP_K = 10
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

chunks = pd.read_csv("data/processed/chunks.csv")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index("data/indexes/dense.index")

while True:
    query = input("\nQuery (or quit): ")
    if query.lower() == "quit":
        break

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, TOP_K)

    print("\nTop Results\n")
    for rank, idx in enumerate(indices[0], start=1):
        row = chunks.iloc[idx]

        print("=" * 70)
        print(f"Rank {rank}")
        print(f"Score: {scores[0][rank-1]:.4f}")
        print(row["title"])
        print(row["url"])
        print()
        print(row["chunk_text"][:500])
        print()
