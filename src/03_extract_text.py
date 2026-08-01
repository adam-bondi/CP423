import pandas as pd
from bs4 import BeautifulSoup
import os

DOCUMENTS_FILE = "data/metadata/documents.csv"
RAW_DIRECTORY = "data/raw/pages"
OUTPUT_PATH = "data/processed/documents_text.csv"

# tags/selectors that are boilerplate, not article content
STRIP_TAGS = ["script", "style", "nav", "footer", "header", "noscript", "form", "aside"]
# CSS selectors commonly used for boilerplate on university sites; adjusted after inspecting a few pages
STRIP_SELECTORS = [".breadcrumb", ".site-header", ".site-footer", "#navigation", ".skip-link"]

MIN_WORD_COUNT = 50  # drop pages that are near empty after cleaning

# function to extract page title and text details
def extract_page(html: str):
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for selector in STRIP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # prefer <main> if present, else fall back to <body>
    main = soup.find("main") or soup.find("body") or soup
    text = main.get_text(separator=" ", strip=True)
    text = " ".join(text.split())  # collapse whitespace

    return title, text

# read metadata and run function on collection corpus
df = pd.read_csv(DOCUMENTS_FILE)
records = []
missing_html = 0
too_short = 0

for _, row in df.iterrows():
    doc_id = row["doc_id"]
    html_path = os.path.join(RAW_DIRECTORY, f"{doc_id}.html")

    if not os.path.exists(html_path):
        missing_html += 1
        continue

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    title, text = extract_page(html)
    word_count = len(text.split())

    if word_count < MIN_WORD_COUNT:
        too_short += 1
        continue

    records.append({
        "doc_id": doc_id,
        "url": row["url"],
        "section": row["section"],
        "last_modified": row["last_modified"],
        "title": title,
        "text": text,
        "word_count": word_count,
    })

# make output directory
os.makedirs("data/processed", exist_ok=True)
out_df = pd.DataFrame(records)
out_df.to_csv(OUTPUT_PATH, index=False)

# summary
print(f"Pages with no HTML on disk: {missing_html}")
print(f"Pages dropped as too short (<{MIN_WORD_COUNT} words): {too_short}")
print(f"Final documents with extracted text: {len(out_df)}")
print(f"Saved to {OUTPUT_PATH}")
