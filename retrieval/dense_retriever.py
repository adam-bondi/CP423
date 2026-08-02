import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# create class to retrieve Dense indexes
class DenseRetriever:

    def __init__(
        self,
        chunks_file="data/processed/chunks.csv",
        index_file="data/indexes/dense.index",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.chunks = pd.read_csv(chunks_file)
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_file)

    def retrieve(self, query, top_k=5):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            embedding,
            top_k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            row = self.chunks.iloc[idx]
            results.append({
                "score": float(score),
                "chunk_id": int(row["chunk_id"]),
                "doc_id": int(row["doc_id"]),
                "title": row["title"],
                "url": row["url"],
                "chunk_text": row["chunk_text"]

            })

        return results
