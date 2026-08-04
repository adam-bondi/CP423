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
        for rank, chunk in enumerate(chunks, start=1):
            context += (
                f"""
        ==============================
        Retrieved Chunk Rank {rank}
        Chunk ID: {chunk['chunk_id']}
        Title: {chunk['title']}
        URL: {chunk['url']}

        Content:
        {chunk['chunk_text']}
        ==============================
        """
            )

        prompt = f"""You are a question answering assistant.

    Retrieved Context:
    {context}

    Question:
    {question}

    Answer the question using ONLY the retrieved context provided above. Do not use outside knowledge.

    If the retrieved context contains enough information to answer the question:
    - Provide a clear answer using only the information from the context.
    - After each statement based on the retrieved context, include the supporting chunk inline citation using this format: [Chunk ID].

    If the retrieved context does not contain enough information to answer the question, respond exactly:
    "I don't know."
    """

        try:
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
        except requests.exceptions.ConnectionError:
            print("Could not connect to Ollama. Make sure 'ollama serve' is running.")
