import pandas as pd
import requests
import os
import time

DOCUMENTS_CSV = "data/metadata/documents.csv"
RAW_DIR = "data/raw/pages"
HEADERS = {"User-Agent": "Mozilla/5.0 (CP423 course RAG project; contact: student@wlu.ca)"}
SLEEP_SECONDS = 0.5  # politeness delay between requests

os.makedirs(RAW_DIR, exist_ok=True)

df = pd.read_csv(DOCUMENTS_CSV)

fetched = 0
skipped = 0
failed = []

for _, row in df.iterrows():
    doc_id = row["doc_id"]
    url = row["url"]
    out_path = os.path.join(RAW_DIR, f"{doc_id}.html")

    # resumable: skip pages already downloaded
    if os.path.exists(out_path):
        skipped += 1
        continue

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(resp.text)
        fetched += 1
    except requests.RequestException as e:
        failed.append({"doc_id": doc_id, "url": url, "error": str(e)})

    time.sleep(SLEEP_SECONDS)

print(f"Fetched: {fetched}")
print(f"Already had (skipped): {skipped}")
print(f"Failed: {len(failed)}")

if failed:
    failed_df = pd.DataFrame(failed)
    os.makedirs("data/metadata", exist_ok=True)
    failed_df.to_csv("data/metadata/fetch_failures.csv", index=False)
    print("Saved failure details to data/metadata/fetch_failures.csv")
