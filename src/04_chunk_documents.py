import pandas as pd
import os

DOCUMENTS_TEXT_CSV = "data/processed/documents_text.csv"
OUTPUT_PATH = "data/processed/chunks.csv"

CHUNK_WORDS = 400      # target chunk size in words
OVERLAP_WORDS = 50     # overlap between consecutive chunks

# function to chunk text
def chunk_text(text: str, chunk_words: int, overlap_words: int):
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    start = 0
    step = chunk_words - overlap_words
    while start < len(words):
        chunk = words[start:start + chunk_words]
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks

# run above function on corpus document text to chunk into 200-word passages with 40-word overlap
df = pd.read_csv(DOCUMENTS_TEXT_CSV)
records = []

for _, row in df.iterrows():
    doc_id = row["doc_id"]
    pieces = chunk_text(str(row["text"]), CHUNK_WORDS, OVERLAP_WORDS)

    for i, piece in enumerate(pieces):
        records.append({
            "chunk_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "chunk_index": i,
            "url": row["url"],
            "section": row["section"],
            "title": row["title"],
            "last_modified": row["last_modified"],
            "chunk_text": piece,
            "word_count": len(piece.split()),
        })

# make output directory
os.makedirs("data/processed", exist_ok=True)
chunks_df = pd.DataFrame(records)
chunks_df.to_csv(OUTPUT_PATH, index=False)

# summary
print(f"Documents processed: {df['doc_id'].nunique()}")
print(f"Total chunks created: {len(chunks_df)}")
print(f"Avg chunks per document: {len(chunks_df) / df['doc_id'].nunique():.2f}")
print(f"Saved to {OUTPUT_PATH}")
