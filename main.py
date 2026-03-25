import os
import time
import requests
from datetime import datetime

from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker
from processing.embedder import EmbeddingGenerator
from retrieval.faiss_index import FAISSIndex
from retrieval.query_processor import QueryProcessor
from retrieval.context_builder import ContextBuilder
from utils.save_embeddings import save_embeddings, load_embeddings


STALE_THRESHOLD_DAYS = 180
TOP_K = 5
MAX_CONTEXT_CHARS = 2500
MAX_CHUNKS = 5


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
        "model": "llama3.2",
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


def check_stale_documents(results):
    """
    Detect outdated documents based on last_updated timestamp.
    Returns list of warnings (unique per source).
    """

    warnings = []
    seen_sources = set()

    current_time = time.time()

    for r in results:
        source = r.get("source")
        last_updated = r.get("last_updated")

        if not source or not last_updated:
            continue

        if source in seen_sources:
            continue

        age_days = (current_time - last_updated) / (60 * 60 * 24)

        if age_days > STALE_THRESHOLD_DAYS:
            readable_time = datetime.fromtimestamp(last_updated).strftime("%Y-%m-%d")

            warnings.append({
                "source": source,
                "age_days": int(age_days),
                "last_updated": readable_time
            })

            seen_sources.add(source)

    return warnings


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
    context_builder = ContextBuilder(
        max_context_chars=MAX_CONTEXT_CHARS,
        max_chunks=MAX_CHUNKS
    )

    while True:
        print("\n" + "=" * 80)
        query = input("Enter your question (or 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Empty query. Try again.")
            continue

        results = query_processor.process_query(query, top_k=TOP_K)

        if not results:
            print("No relevant results found.")
            continue

        if results[0]["similarity"] < 0.2:
            print("Low confidence retrieval. Answer may be unreliable.\n")

        stale_warnings = check_stale_documents(results)

        if stale_warnings:
            print("\nPotentially outdated documentation detected:")
            for w in stale_warnings:
                print(
                    f"- {w['source']} (Last updated: {w['last_updated']}, "
                    f"{w['age_days']} days old)"
                )
            print()

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