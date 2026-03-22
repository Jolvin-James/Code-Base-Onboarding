# retrieval/faiss_index.py
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    Production-ready FAISS index handler.

    Features:
    - Index creation
    - Vector addition
    - Similarity search
    - Metadata mapping
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: embedding size (e.g., 384)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[Dict] = []

        logger.info(f"FAISS Index initialized with dimension={dimension}")

    # ADD VECTORS
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Add embeddings to FAISS index.

        Args:
            embeddings: np.ndarray (num_chunks, dim)
            chunks: metadata list
        """

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks size mismatch")

        if embeddings.dtype != np.float32:
            raise TypeError("Embeddings must be float32 for FAISS")

        logger.info(f"Adding {len(embeddings)} vectors to FAISS index")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

        logger.info("Vectors added successfully")

    # SEARCH
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for most similar chunks.

        Args:
            query_vector: np.ndarray (1, dim)
            top_k: number of results

        Returns:
            List of top matching chunks with scores
        """

        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)

        distances, indices = self.index.search(query_vector, top_k)

        results = []

        for rank, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            chunk = self.chunks[idx].copy()

            results.append({
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "content": chunk["content"],
                "header": chunk.get("header"),
                "source": chunk.get("source"),
                "last_updated": chunk.get("last_updated"),
            })

        return results

    # INFO
    def get_size(self) -> int:
        return self.index.ntotal