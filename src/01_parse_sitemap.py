import xml.etree.ElementTree as ET
import pandas as pd

SITEMAP = "data/raw/sitemap.xml"

# XML namespace used by sitemap.xml
ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

tree = ET.parse(SITEMAP)
root = tree.getroot()

documents = []

doc_id = 1

for url in root.findall("sm:url", ns):

    loc = url.find("sm:loc", ns).text
    lastmod = url.find("sm:lastmod", ns)

    lastmod = lastmod.text if lastmod is not None else ""

    # extract section from URL
    path = loc.replace("https://students.wlu.ca/", "")

    section = path.split("/")[0]

    documents.append({
        "doc_id": doc_id,
        "url": loc,
        "section": section,
        "last_modified": lastmod
    })

    doc_id += 1

df = pd.DataFrame(documents)

print(df.head())
print()
print(df["section"].value_counts())

df.to_csv("data/metadata/documents.csv", index=False)

print(f"\nSaved {len(df)} documents.")