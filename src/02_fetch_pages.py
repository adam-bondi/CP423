import pandas as pd
import requests
import os
import time

DOCUMENTS_FILE = "data/metadata/documents.csv"
OUTPUT_DIR = "data/raw/pages"

USER_AGENT = {"User-Agent": "CP423-RAG-Project/1.0 (Educational Web Crawler; Wilfrid Laurier University)"}
SLEEP_SECONDS = 0.5  # seconds delay between requests
REQUEST_TIMEOUT = 10 # seconds

# create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# read metadata
documents = pd.read_csv(DOCUMENTS_FILE)

# download pages
fetched = 0
skipped = 0
failed = []

for _, row in documents.iterrows():
    doc_id = row["doc_id"]
    url = row["url"]
    output_path = os.path.join(OUTPUT_DIR, f"{doc_id:04d}.html")

    # skip pages already downloaded
    if os.path.exists(output_path):
        skipped += 1
        continue

    try:
        response = requests.get(url, headers=USER_AGENT, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)   
        fetched += 1
    except requests.RequestException as e:
        failed.append({"doc_id": doc_id, "url": url, "error": str(e)})

    time.sleep(SLEEP_SECONDS)

# summary
print(f"Fetched: {fetched}")
print(f"Already downloaded (skipped): {skipped}")
print(f"Failed: {len(failed)}")

# save failed downloads
if failed:
    failed_df = pd.DataFrame(failed)
    os.makedirs("data/metadata", exist_ok=True)
    failed_df.to_csv("data/metadata/fetch_failures.csv", index=False)
    print("Saved failure details to data/metadata/fetch_failures.csv")
