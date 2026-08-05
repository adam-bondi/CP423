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

    chunks = retriever.retrieve(question, top_k=7)
    print("\nRetrieved Chunks\n")

    for rank, chunk in enumerate(chunks, start=1):
        print("=" * 70)
        print(f"Rank: {rank}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Title: {chunk['title']}")
        print(f"URL: {chunk['url']}")
        print("\nContent:")
        print(f"{chunk['chunk_text'][:250]}...")
        print()

    print("\nGenerating answer...\n")
    answer = generator.generate(
        question,
        chunks
    )

    print(answer)
