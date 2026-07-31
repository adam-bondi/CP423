import xml.etree.ElementTree as ET
import pandas as pd
import os

SITEMAP = "data/raw/sitemap.xml"

# URL patterns to exclude from the retrieval corpus
EXCLUDED_PATTERNS = {
    "/news/", "/search/", "sitemap.html"
}

# XML namespace used by sitemap.xml
ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# read sitemap
tree = ET.parse(SITEMAP)
root = tree.getroot()

documents = []
excluded_count = 0

for url in root.findall("sm:url", ns):
    
    loc = url.find("sm:loc", ns).text
    lastmod = url.find("sm:lastmod", ns)
    lastmod = lastmod.text if lastmod is not None else ""

    if any(pattern in loc for pattern in EXCLUDED_PATTERNS):
        excluded_count += 1
        continue

    # extract section from URL
    path = loc.replace("https://students.wlu.ca/", "")
    parts = path.split("/")
    if len(parts)==1:
        section="home"
    else:  
        section = parts[0]

    documents.append({
        "url": loc,
        "section": section,
        "last_modified": lastmod
    })

df = pd.DataFrame(documents)
df.insert(0, "doc_id", range(1, len(df)+1))


print(df.head())
print()

print("Documents by section:")
print(df["section"].value_counts())

print()
print(f"Excluded pages: {excluded_count}")
print(f"Final corpus size: {len(df)}")

os.makedirs("data/metadata", exist_ok=True)
df.to_csv("data/metadata/documents.csv", index=False)

print(f"\nSaved documents.csv")
