import os
import time
import requests
from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker
from processing.embedder import EmbeddingGenerator
from retrieval.faiss_index import FAISSIndex
from retrieval.query_processor import QueryProcessor
from retrieval.context_builder import ContextBuilder
from utils.save_embeddings import save_embeddings, load_embeddings

def build_prompt(context, query):
    system_prompt = """
You are a strict codebase assistant.

RULES:
- Answer ONLY using the provided context
- Do NOT guess
- If not found, say: "I could not find this in the documentation."
- Keep answers short and precise
"""

    user_prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

    return system_prompt, user_prompt

def call_llm(system_prompt, user_prompt):
    url = "http://localhost:11434/api/generate"

    full_prompt = f"""
        {system_prompt}

        {user_prompt}
    """

    payload = {
        "model": "llama3.2",   # or mistral, phi3, etc.
        "prompt": full_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            return f"Ollama Error: {response.text}"

        data = response.json()

        return data.get("response", "No response from model.")

    except Exception as e:
        return f"Local LLM Error: {str(e)}"


def main():
    start_time = time.time()

    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

    print(f"\nLoading documents from {ROOT_DIR}...")

    try:
        documents = load_documents(ROOT_DIR)
    except FileNotFoundError:
        print(f"Error: Directory '{ROOT_DIR}' not found.")
        return

    if not documents:
        print("No documents found. Exiting.")
        return

    print(f"Loaded {len(documents)} documents")

    print("\nChunking documents...")
    chunker = MarkdownChunker(min_chunk_size=100)
    chunks = chunker.chunk_documents(documents)

    if not chunks:
        print("No chunks generated. Exiting.")
        return

    print(f"Generated {len(chunks)} chunks")

    embedder = EmbeddingGenerator(normalize=True)

    try:
        embeddings, chunks = load_embeddings()
        print("\nLoaded cached embeddings")
    except Exception:
        print("\nGenerating embeddings...")
        embeddings, _ = embedder.embed_chunks(
            chunks,
            batch_size=32,
            return_chunks=False
        )
        save_embeddings(embeddings, chunks)
        print("Embeddings generated and saved")

    print("\nCreating FAISS index...")
    dimension = embeddings.shape[1]

    faiss_index = FAISSIndex(dimension)
    faiss_index.add_embeddings(embeddings, chunks)

    print(f"FAISS index size: {faiss_index.get_size()}")

    query_processor = QueryProcessor(embedder, faiss_index)
    context_builder = ContextBuilder(max_context_chars=2500, max_chunks=5)

    while True:
        print("\n" + "=" * 80)
        query = input("Enter your question (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Empty query. Try again.")
            continue

        results = query_processor.process_query(query, top_k=5)

        if not results:
            print("No relevant results found.")
            continue

        # Low confidence warning
        if results[0]["similarity"] < 0.2:
            print("⚠️ Low confidence retrieval. Answer may be unreliable.\n")

        context = context_builder.build_context(results)

        print("\n" + "-" * 80)
        print("CONTEXT SENT TO LLM")
        print("-" * 80)
        print(context)
        print("-" * 80)

        system_prompt, user_prompt = build_prompt(context, query)

        response = call_llm(system_prompt, user_prompt)

        print("\n" + "=" * 80)
        print("FINAL ANSWER")
        print("=" * 80)
        print(response)

    end_time = time.time()
    print(f"\nExecution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()