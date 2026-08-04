import requests

class OllamaGenerator:

    def __init__(
        self,
        model="llama3.2",
        temperature=0
    ):

        self.model = model
        self.temperature = temperature

    def generate(self, question, chunks):

        context = ""
        for chunk in chunks:

            context += (
                f"[Chunk {chunk['chunk_id']}]\n"
                f"{chunk['chunk_text']}\n\n"
            )

        prompt = f"""You are a question answering assistant that answers ONLY using the retrieved context below.

Do NOT use any outside knowledge, even if you are confident you know the correct answer. Your only source of truth is the context provided.

If the context does not contain the answer, respond with EXACTLY these three words and nothing else:
I don't know.

If the context DOES contain the answer, answer using only that information, and end every sentence that uses context information with a citation in the format [Chunk ID].

Retrieved Context
{context}

Question: {question}

Remember: use ONLY the context above, and cite every fact you use as [Chunk ID], this is most important. If the context doesn't contain the answer, respond with exactly "I don't know." and nothing else.
"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature}
            }
        )

        return response.json()["response"]
