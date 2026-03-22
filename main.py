from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker
from processing.embedder import EmbeddingGenerator

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

    print("\nEmbedding generation complete")
    print("Shape:", embeddings.shape)
    print("Dtype:", embeddings.dtype)

if __name__ == "__main__":
    main()