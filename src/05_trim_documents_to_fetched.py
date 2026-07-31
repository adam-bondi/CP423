import pandas as pd
import os


df = pd.read_csv("data/metadata/documents.csv")
fetched_ids = {int(f.split(".")[0]) for f in os.listdir("data/raw/pages") if f.endswith(".html")}
df = df[df["doc_id"].isin(fetched_ids)]
df.to_csv("data/metadata/documents.csv", index=False)
print(f"Trimmed to {len(df)} documents")
