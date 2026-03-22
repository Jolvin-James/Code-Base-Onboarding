# processing/embedder.py

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Production-ready embedding generator using SentenceTransformers.

    Features:
    - Batch encoding
    - NumPy conversion (FAISS compatible)
    - Optional normalization
    - Logging + validation
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: SentenceTransformer model
            normalize: Whether to L2 normalize embeddings
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        logger.info(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize

        logger.info("Model loaded successfully")

    # -----------------------------
    # MAIN EMBEDDING FUNCTION
    # -----------------------------
    def embed_chunks(
        self,
        chunks: List[Dict],
        batch_size: int = 32,
        return_chunks: bool = False
    ) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        Generate embeddings for chunk list.

        Args:
            chunks: List of chunk dicts
            batch_size: Batch size for encoding
            return_chunks: Attach embeddings to chunks

        Returns:
            embeddings: np.ndarray (num_chunks, dim)
            chunks (optional): with embeddings attached
        """

        if not chunks:
            raise ValueError("No chunks provided for embedding.")

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        texts = [
            self._safe_text(
                f"{chunk.get('header', '')}\n{chunk.get('content', '')}"
            )
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # we handle manually
        )

        # Ensure float32 (FAISS requirement)
        embeddings = embeddings.astype(np.float32)

        # Normalize if enabled
        if self.normalize:
            embeddings = self._normalize(embeddings)

        self._validate_embeddings(embeddings, len(chunks))

        logger.info(f"Embeddings generated: shape={embeddings.shape}")

        if return_chunks:
            enriched_chunks = self._attach_embeddings(chunks, embeddings)
            return embeddings, enriched_chunks

        return embeddings, None

    # -----------------------------
    # QUERY EMBEDDING
    # -----------------------------
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype(np.float32)

        if self.normalize:
            embedding = self._normalize(embedding)

        return embedding

    # -----------------------------
    # INTERNAL UTILITIES
    # -----------------------------
    def _safe_text(self, text: str) -> str:
        """
        Clean and validate text input.
        """
        if not isinstance(text, str):
            return ""

        return text.strip()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalize vectors.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        # Avoid division by zero
        norms[norms == 0] = 1e-10

        return vectors / norms

    def _validate_embeddings(self, embeddings: np.ndarray, expected_len: int):
        """
        Validate embeddings integrity.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a NumPy array.")

        if embeddings.shape[0] != expected_len:
            raise ValueError("Mismatch between chunks and embeddings.")

        if embeddings.dtype != np.float32:
            raise TypeError("Embeddings must be float32.")

    def _attach_embeddings(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Attach embeddings to chunk metadata (optional).
        """
        enriched = []

        for chunk, emb in zip(chunks, embeddings):
            new_chunk = chunk.copy()
            new_chunk["embedding"] = emb
            enriched.append(new_chunk)

        return enriched