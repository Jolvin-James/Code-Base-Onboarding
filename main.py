import os
from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker
from processing.embedder import EmbeddingGenerator
from retrieval.faiss_index import FAISSIndex
from retrieval.query_processor import QueryProcessor


def main():
    # Ensure ROOT_DIR resolves correctly regardless of where the script is run from
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

    print("\nChunking documents...")
    chunker = MarkdownChunker(min_chunk_size=100)
    chunks = chunker.chunk_documents(documents)

    if not chunks:
        print("No chunks were generated. Exiting.")
        return

    print(f"Generated {len(chunks)} chunks")

    print("\nGenerating embeddings...")
    embedder = EmbeddingGenerator(normalize=True)

    embeddings, _ = embedder.embed_chunks(
        chunks,
        batch_size=32,
        return_chunks=False
    )

    print("\nCreating FAISS index...")
    dimension = embeddings.shape[1]

    faiss_index = FAISSIndex(dimension)
    faiss_index.add_embeddings(embeddings, chunks)

    print(f"FAISS index size: {faiss_index.get_size()}")

    # TEST SEARCH
    print("\nTesting search with QueryProcessor...\n")

    query_processor = QueryProcessor(embedder, faiss_index)

    query = "How authentication works"

    results = query_processor.process_query(query, top_k=8)

    for r in results:
        print(f"\nRank: {r['rank']}")
        print(f"Similarity: {r['similarity']:.4f}")
        print(f"Header: {r['header']}")
        print(f"Source: {r['source']}")
        print("-" * 50)


if __name__ == "__main__":
    main()