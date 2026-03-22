from ingestion.loader import load_documents
from processing.chunker import MarkdownChunker

docs = load_documents("docs")

chunker = MarkdownChunker(min_chunk_size=100)
chunks = chunker.chunk_documents(docs)

print(f"Total chunks created: {len(chunks)}")

# Example output
print(chunks[0])