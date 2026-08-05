import pandas as pd

RESULTS_PATH = "data/evaluation/evaluation_results.csv"
GRADES_PATH = "data/evaluation/graded_columns_top7.csv"

results = pd.read_csv(RESULTS_PATH)
grades = pd.read_csv(GRADES_PATH)

# drop the empty grading columns from results before merging in the real ones
results = results.drop(columns=["correct", "supported", "citation_correct", "idk_correct"])

merged = results.merge(
    grades,
    on=["question_id", "retriever"],
    how="left"
)

merged.to_csv(RESULTS_PATH, index=False)
print(f"Merged grades into {RESULTS_PATH}")
print(f"Rows: {len(merged)}")
print(merged[["question_id", "retriever", "correct", "supported", "citation_correct", "idk_correct"]])