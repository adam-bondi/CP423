import pandas as pd

RESULTS = "data/evaluation/evaluation_results.csv"

df = pd.read_csv(RESULTS)

# convert YES/NO strings
def yes(x):
    return str(x).strip().upper() == "YES"

# summarize retrievers
for retriever in ["BM25", "Dense"]:

    print("\n" + "=" * 70)
    print(retriever)
    print("=" * 70)

    subset = df[df["retriever"] == retriever]
    answerable = subset[subset["type"] != "unanswerable"]
    unanswerable = subset[subset["type"] == "unanswerable"]

    hit_rate = answerable["retrieval_hit"].mean()
    generation_accuracy = answerable["correct"].apply(yes).mean()
    supported = answerable["supported"].apply(yes).mean()
    citation_accuracy = answerable["citation_correct"].apply(yes).mean()
    idk_accuracy = unanswerable["idk_correct"].apply(yes).mean()

    print(f"Questions: {len(subset)}")
    print(f"Answerable Questions: {len(answerable)}")
    print(f"Unanswerable Questions: {len(unanswerable)}")
    print()

    print(f"Hit@5: {hit_rate:.2%}")
    print(f"Generation Accuracy: {generation_accuracy:.2%}")
    print(f"Supported Answers: {supported:.2%}")
    print(f"Citation Accuracy: {citation_accuracy:.2%}")
    print(f'I don\'t know Accuracy: {idk_accuracy:.2%}')