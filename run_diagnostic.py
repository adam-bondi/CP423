import pandas as pd
import requests
import json
import os

QUESTIONS_CSV = "data/evaluation/diagnostic_questions.csv"
OUT_PATH = "data/evaluation/diagnostic_results.csv"

# Assumes Ollama is running locally with this model pulled (e.g. `ollama pull llama3.2`)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"


def ask_llm_no_context(question: str) -> str:
    prompt = (
        "Answer the following question directly and concisely, "
        "using only your own knowledge.\n\n"
        f"Question: {question}\nAnswer:"
    )
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


df = pd.read_csv(QUESTIONS_CSV)

results = []
for _, row in df.iterrows():
    print(f"Asking Q{row['question_id']}...")
    answer = ask_llm_no_context(row["question"])
    results.append({
        "question_id": row["question_id"],
        "question": row["question"],
        "reference_answer": row["reference_answer"],
        "model_answer": answer,
        "correct": "",  # fill in manually after reading model_answer: yes / no / partial
    })

os.makedirs("data/evaluation", exist_ok=True)
out_df = pd.DataFrame(results)
out_df.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(out_df)} results to {OUT_PATH}")
print("Open the CSV and fill in the 'correct' column (yes/no/partial) by comparing model_answer to reference_answer.")
