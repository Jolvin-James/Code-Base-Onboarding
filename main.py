import os
import time

from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker
from processing.embedder import EmbeddingGenerator
from retrieval.faiss_index import FAISSIndex
from retrieval.query_processor import QueryProcessor
from retrieval.context_builder import ContextBuilder
from utils.save_embeddings import save_embeddings, load_embeddings


def main():
    start_time = time.time()

    # Ensure ROOT_DIR resolves correctly regardless of execution location
    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

    print(f"\nLoading documents from {ROOT_DIR}...")
    try:
        documents = load_documents(ROOT_DIR)
    except FileNotFoundError:
        print(f"Error: Directory '{ROOT_DIR}' not found. Please create it and add markdown files.")
        return

    if not documents:
        print("No documents found to process. Exiting.")
        return

    print(f"Loaded {len(documents)} documents")

    # CHUNKING
    print("\nChunking documents...")
    chunker = MarkdownChunker(min_chunk_size=100)
    chunks = chunker.chunk_documents(documents)

    if not chunks:
        print("No chunks were generated. Exiting.")
        return

    print(f"Generated {len(chunks)} chunks")

    # EMBEDDINGS (WITH CACHING)
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

    # FAISS INDEX
    print("\nCreating FAISS index...")
    dimension = embeddings.shape[1]

    faiss_index = FAISSIndex(dimension)
    faiss_index.add_embeddings(embeddings, chunks)

    print(f"FAISS index size: {faiss_index.get_size()}")

    # QUERY + CONTEXT PIPELINE
    query_processor = QueryProcessor(embedder, faiss_index)
    context_builder = ContextBuilder(max_context_chars=2500, max_chunks=5)

    test_queries = [
        "How authentication works",
        "jwt validation flow",
        "non existing feature xyz"
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        if not query.strip():
            print("Empty query. Skipping...")
            continue

        results = query_processor.process_query(query, top_k=5)

        if not results:
            print("No relevant results found.")
            continue

        # Low confidence warning
        if results[0]["similarity"] < 0.2:
            print("⚠️ Low confidence retrieval. Results may be inaccurate.\n")

        # Build context
        context = context_builder.build_context(results)

        # Display context
        print("\n" + "-" * 80)
        print("CONTEXT SENT TO LLM")
        print("-" * 80)
        print(context)
        print("-" * 80)

        # Display ranked results (top 5)
        print("\nTop Retrieved Chunks:")
        for r in results[:5]:
            print(f"\nRank: {r['rank']}")
            print(f"Similarity: {r['similarity']:.4f}")
            print(f"Header: {r['header']}")
            print(f"Source: {r['source']}")
            print("-" * 40)

    end_time = time.time()
    print(f"\nExecution Time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()