import os
import json
import csv
import time
import requests
from typing import Dict, List

# Assuming these are your custom local modules
from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker
from processing.embedder import EmbeddingGenerator
from retrieval.faiss_index import FAISSIndex
from retrieval.query_processor import QueryProcessor
from retrieval.context_builder import ContextBuilder
from utils.save_embeddings import load_embeddings, save_embeddings

DATASET_PATH = "evaluation/golden_dataset.json"
OUTPUT_CSV = "evaluation/evaluation_results.csv"
TOP_K = 5
MAX_CONTEXT_CHARS = 2500
MAX_CHUNKS = 5
SIMILARITY_THRESHOLD = 0.2

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
            return f"LLM Error: {response.text}"

        return response.json().get("response", "")

    except Exception as e:
        return f"LLM Exception: {str(e)}"

def build_prompt(context, query):
    system_prompt = """
You are a strict codebase assistant.
RULES:

Answer ONLY using the provided context
Do NOT guess
If not found, say: "I could not find this in the documentation."
Keep answers short and precise
"""
    user_prompt = f"""
Context:
{context}
Question:
{query}
Answer:
"""
    return system_prompt, user_prompt

def evaluate_retrieval(results: List[Dict], expected_source: str) -> int:
    """
    Check if correct source is in top_k results
    """
    if expected_source == "none":
        return 1 # not applicable
    
    for r in results:
        if expected_source in r.get("source", ""):
            return 1
    return 0

def evaluate_correctness(answer: str, expected_answer: str) -> int:
    """
    Simple substring match (can be improved later)
    """
    return int(expected_answer.lower() in answer.lower())

def evaluate_faithfulness(answer: str, context: str) -> int:
    """
    Check if answer content appears in context
    """
    return int(any(word in context.lower() for word in answer.lower().split()[:5]))

def evaluate_safety(answer: str, expected_source: str) -> int:
    """
    Ensure model says 'not found' for out-of-scope
    """
    if expected_source == "none":
        return int("not find" in answer.lower() or "could not find" in answer.lower())
    return 1

def main():
    start_time = time.time()
    print("Loading golden dataset...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

    print("Loading documents...")
    documents = load_documents(ROOT_DIR)

    print("Chunking...")
    chunker = MarkdownChunker(min_chunk_size=100)
    chunks = chunker.chunk_documents(documents)

    embedder = EmbeddingGenerator(normalize=True)

    try:
        embeddings, chunks = load_embeddings()
        print("Loaded cached embeddings")
    except Exception:
        print("Generating embeddings...")
        embeddings, _ = embedder.embed_chunks(chunks)
        save_embeddings(embeddings, chunks)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    faiss_index = FAISSIndex(dimension)
    faiss_index.add_embeddings(embeddings, chunks)

    query_processor = QueryProcessor(embedder, faiss_index)
    context_builder = ContextBuilder(
        max_context_chars=MAX_CONTEXT_CHARS,
        max_chunks=MAX_CHUNKS
    )

    results_rows = []

    print("\nStarting evaluation...\n")

    for i, item in enumerate(dataset):
        query = item["query"]
        expected_answer = item["expected_answer"]
        expected_source = item["source"]

        print(f"[{i+1}/{len(dataset)}] {query}")

        results = query_processor.process_query(query, top_k=TOP_K)
        context = context_builder.build_context(results)

        system_prompt, user_prompt = build_prompt(context, query)
        answer = call_llm(system_prompt, user_prompt)

        retrieval_score = evaluate_retrieval(results, expected_source)
        correctness_score = evaluate_correctness(answer, expected_answer)
        faithfulness_score = evaluate_faithfulness(answer, context)
        safety_score = evaluate_safety(answer, expected_source)

        final_score = (
            retrieval_score +
            correctness_score +
            faithfulness_score +
            safety_score
        ) / 4

        results_rows.append({
            "query": query,
            "expected_source": expected_source,
            "retrieval": retrieval_score,
            "correctness": correctness_score,
            "faithfulness": faithfulness_score,
            "safety": safety_score,
            "final_score": round(final_score, 2)
        })

    if not results_rows:
        print("No results to save. Dataset might be empty.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)

    print("\nEvaluation complete!")
    print(f"Results saved to: {OUTPUT_CSV}")

    avg_score = sum(r["final_score"] for r in results_rows) / len(results_rows)
    print(f"\nAverage Score: {avg_score:.2f}")

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()