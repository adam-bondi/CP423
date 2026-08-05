import pandas as pd

RESULTS = "data/evaluation/evaluation_results.csv"
OUTPUT = "data/evaluation/metrics.txt"

df = pd.read_csv(RESULTS)

# convert YES/NO strings
def yes(x):
    return str(x).strip().upper() == "YES"

lines = []

# summarize retrievers
for retriever in ["BM25", "Dense"]:

    subset = df[df["retriever"] == retriever]
    answerable = subset[subset["type"] != "unanswerable"]
    unanswerable = subset[subset["type"] == "unanswerable"]

    hit_rate = answerable["retrieval_hit"].mean()
    generation_accuracy = answerable["correct"].apply(yes).mean()
    supported = answerable["supported"].apply(yes).mean()
    citation_accuracy = answerable["citation_correct"].apply(yes).mean()
    idk_accuracy = unanswerable["idk_correct"].apply(yes).mean()

    lines.append("=" * 70)
    lines.append(retriever)
    lines.append("=" * 70)
    lines.append(f"Questions: {len(subset)}")
    lines.append(f"Answerable Questions: {len(answerable)}")
    lines.append(f"Unanswerable Questions: {len(unanswerable)}")
    lines.append("")
    lines.append(f"Hit@5: {hit_rate:.2%}")
    lines.append(f"Generation Accuracy: {generation_accuracy:.2%}")
    lines.append(f"Supported Answers: {supported:.2%}")
    lines.append(f"Citation Accuracy: {citation_accuracy:.2%}")
    lines.append(f"I don't know Accuracy: {idk_accuracy:.2%}")
    lines.append("")

# write metrics to file
with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Metrics written to {OUTPUT}")
