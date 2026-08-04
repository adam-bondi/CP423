import pickle
import pandas as pd
import nltk

# create class to retrieve BM25 indexes
class BM25Retriever:
    def __init__(
        self,
        chunks_file="data/processed/chunks.csv",
        index_file="data/indexes/bm25_index.pkl"
    ):

        self.chunks = pd.read_csv(chunks_file)
        with open(index_file, "rb") as f:
            self.bm25 = pickle.load(f)

    def retrieve(self, query, top_k=5):
        tokens = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        top = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top:
            row = self.chunks.iloc[idx]
            results.append({
                "score": float(scores[idx]),
                "chunk_id": str(row["chunk_id"]),
                "doc_id": int(row["doc_id"]),
                "title": row["title"],
                "url": row["url"],
                "chunk_text": row["chunk_text"]
            })

        return results
