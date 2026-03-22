from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker
from processing.embedder import EmbeddingGenerator
from retrieval.faiss_index import FAISSIndex


def main():
    ROOT_DIR = "docs"

    print("\nLoading documents...")
    documents = load_documents(ROOT_DIR)

    print(f"Loaded {len(documents)} documents")

    print("\nChunking documents...")
    chunker = MarkdownChunker(min_chunk_size=100)
    chunks = chunker.chunk_documents(documents)

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
    print("\nTesting search...\n")

    query = "How authentication works"
    query_vector = embedder.embed_query(query)

    results = faiss_index.search(query_vector, top_k=8)

    for r in results:
        print(f"\nRank: {r['rank']}")
        print(f"Score: {r['score']}")
        print(f"Header: {r['header']}")
        print(f"Source: {r['source']}")
        print("-" * 50)


if __name__ == "__main__":
    main()