import json
import os
import pandas as pd

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from generation.ollama_generator import OllamaGenerator


# configuration
QUESTIONS_FILE = "data/evaluation/gold_questions.csv"
OUTPUT_FILE = "data/evaluation/evaluation_results.csv"

TOP_K = 5

# load evaluation set
questions = pd.read_csv(QUESTIONS_FILE)

retrievers = {
    "BM25": BM25Retriever(),
    "Dense": DenseRetriever()
}

generator = OllamaGenerator(
    model="llama3.2",
    temperature=0
)

results = []
# run evaluation
for retriever_name, retriever in retrievers.items():

    print(f"\n===== Evaluating {retriever_name} =====\n")
    for _, row in questions.iterrows():

        question_id = row["question_id"]
        question = row["question"]
        question_type = row["type"]
        reference_answer = row["reference_answer"]
        print(f"{retriever_name} - {question_id}")

        # parse ground truth chunk ids
        ground_truth = json.loads(row["ground_truth_chunk_ids"])

        # retrieve chunks
        retrieved_chunks = retriever.retrieve(
            question,
            top_k=TOP_K
        )

        retrieved_ids = [
            chunk["chunk_id"]
            for chunk in retrieved_chunks
        ]

        # unanswerable questions have no ground-truth chunks
        if len(ground_truth) == 0:
            retrieval_hit = None
        else:
            retrieval_hit = any(
                gt in retrieved_ids
                for gt in ground_truth
            )

        # generate answer
        answer = generator.generate(
            question,
            retrieved_chunks
        )

        # save results
        results.append({
            "question_id": question_id,
            "retriever": retriever_name,
            "question": question,
            "type": question_type,
            "ground_truth_chunk_ids": json.dumps(ground_truth),
            "retrieved_chunk_ids": json.dumps(retrieved_ids),
            "retrieval_hit": retrieval_hit,
            "reference_answer": reference_answer,
            "generated_answer": answer,

            # fill  in manually
            "correct": "",
            "supported": "",
            "citation_correct": "",
            "idk_correct": ""
        })


# save CSV
os.makedirs("data/evaluation", exist_ok=True)
results = pd.DataFrame(results)
results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n=====================================")
print("Evaluation complete.")
print(f"Saved {len(results)} rows.")
print(f"Output: {OUTPUT_FILE}")