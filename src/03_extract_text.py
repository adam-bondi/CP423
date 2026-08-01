import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

RAW_DIRECTORY = "data/raw/pages"
METADATA_FILE = "data/metadata/documents.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "documents_text.csv")

MIN_WORDS = 50
# make output file and read metadata
os.makedirs(OUTPUT_DIR, exist_ok=True)
metadata = pd.read_csv(METADATA_FILE)

records = []
# open collection pages doc-by-doc
for _, row in tqdm(metadata.iterrows(),
                   total=len(metadata),
                   desc="Extracting text"):

    doc_id = row["doc_id"]
    html_file = os.path.join(
        RAW_DIRECTORY, f"{doc_id:04d}.html")

    if not os.path.exists(html_file):
        continue

    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    
    # remove elements that are not useful for RAG
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # remove common WLU HTML components
    remove_selectors = [
        ".breadcrumbs", ".breadcrumb", ".pathways", ".gradient", ".image", ".full-width-banner", ".site_header", ".site_footer"
    ]

    for selector in remove_selectors:
        for tag in soup.select(selector):
            tag.decompose()
    
    # focus on <main> content; if not found then <body>
    main = soup.find("main")
    if main is None:
        main = soup.find("body")

    if main is None:
        continue

    # find title and h1
    page_title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = main.find("h1")
    heading = h1.get_text(" ", strip=True) if h1 else ""
    h1 = main.find("h1")
    
    # extract text
    text = main.get_text(separator="\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        continue

    records.append({
        "doc_id": row["doc_id"],
        "title": page_title,
        "heading": heading,
        "url": row["url"],
        "section": row["section"],
        "last_modified": row["last_modified"],
        "word_count": word_count,
        "text": text
    })

# save to CSV output file
documents = pd.DataFrame(records)
documents.to_csv(OUTPUT_FILE, index=False)

# summary
print(f"\nSaved {len(documents)} documents.")
print(f"Output: {OUTPUT_FILE}")
