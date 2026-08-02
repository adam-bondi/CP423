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

        prompt = f"""
You are a question answering assistant.

Answer ONLY using the retrieved context.

If the answer cannot be determined from the context, reply exactly:

"I don't know."

Whenever you use information from a chunk,
cite it inline like [Chunk 17].

Retrieved Context

{context}

Question

{question}
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
