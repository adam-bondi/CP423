import os
import time
import requests
import pandas as pd
from tqdm import tqdm

DOCUMENTS_FILE = "data/metadata/documents.csv"
OUTPUT_DIR = "data/raw/html"
LOG_DIR = "data/logs"

REQUEST_DELAY = 0.2 # seconds between requests
REQUEST_TIMEOUT = 15 # seconds

USER_AGENT = (
    "CP423-RAG-Project/1.0 "
    "(Educational Web Crawler; Wilfrid Laurier University)"
)

# output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# read metadata
documents = pd.read_csv(DOCUMENTS_FILE)
print(f"Found {len(documents)} documents to download.\n")

headers = {
    "User-Agent": USER_AGENT
}

failed_downloads = []

# download pages
for _, row in tqdm(documents.iterrows(),
                   total=len(documents),
                   desc="Downloading"):

    doc_id = int(row["doc_id"])
    url = row["url"]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{doc_id:04d}.html"
    )

    # skip files that already exist
    if os.path.exists(output_file):
        continue

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)

    except Exception as e:

        failed_downloads.append({
            "doc_id": doc_id,
            "url": url,
            "error": str(e)
        })

    time.sleep(REQUEST_DELAY)

# save failed downloads
failed_df = pd.DataFrame(failed_downloads)
failed_df.to_csv(
    os.path.join(LOG_DIR, "failed_downloads.csv"),
    index=False
)

# summary
successful = len(documents) - len(failed_downloads)

print("\n----------------------------------------")
print("Download Complete")
print("----------------------------------------")
print(f"Successful downloads : {successful}")
print(f"Failed downloads     : {len(failed_downloads)}")
print(f"Saved HTML pages to  : {OUTPUT_DIR}")

if len(failed_downloads) > 0:
    print(f"Failure log saved to : {LOG_DIR}/failed_downloads.csv")