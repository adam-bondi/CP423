import pickle
import pandas as pd
import nltk

TOP_K = 10

chunks = pd.read_csv("data/processed/chunks.csv")
with open("data/indexes/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

while True:
    query = input("\nQuery (or 'quit'): ")
    if query.lower() == "quit":
        break

    tokens = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokens)
    top = scores.argsort()[-TOP_K:][::-1]

    print("\nTop Results\n")
    for rank, idx in enumerate(top, start=1):

        row = chunks.iloc[idx]
        print("=" * 70)
        print(f"Rank {rank}")
        print(f"Score: {scores[idx]:.4f}")
        print(row["title"])
        print(row["url"])
        print()
        print(row["chunk_text"][:500])
        print()
