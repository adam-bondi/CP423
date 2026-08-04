from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from generation.ollama_generator import OllamaGenerator


#########################################
# CHANGE ONLY THIS LINE DEPENDING ON RETRIEVER USED
#########################################
print("Using BM25 Retriever")
retriever = BM25Retriever()

# print("Using Dense Retriever")
# retriever = DenseRetriever() 

generator = OllamaGenerator()
while True:

    question = input("\nQuestion (or quit): ")
    if question.lower() == "quit":
        break

    chunks = retriever.retrieve(question, top_k=10)
    print("\nRetrieved Chunks\n")
    for chunk in chunks:
        print("=" * 70)
        print(f"Chunk {chunk['chunk_id']}")
        print(chunk["title"])
        print(chunk["url"])
        print()
        print(chunk["chunk_text"][:250])
        print()

    print("\nGenerating answer...\n")
    answer = generator.generate(
        question,
        chunks
    )

    print(answer)
