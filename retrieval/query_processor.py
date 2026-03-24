from typing import List, Dict
import numpy as np

class QueryProcessor:
    """
    Handles:
    - Query embedding
    - FAISS search
    - Result ranking
    - Formatting
    """

    def __init__(self, embedder, faiss_index):
        self.embedder = embedder
        self.faiss_index = faiss_index

    def process_query(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Full query pipeline.
        """

        # 1. Embed Query
        query_vector = self.embedder.embed_query(query)

        # 2. Search FAISS
        raw_results = self.faiss_index.search(query_vector, top_k=top_k)

        # 3. Re-rank + format
        ranked_results = self._rank_results(raw_results)

        return ranked_results

    # Ranking Logic
    def _rank_results(self, results: List[Dict]) -> List[Dict]:
        """
        Convert L2 distance to similarity score
        """

        for r in results:
            distance = r["score"]

            # Convert distance to similarity
            similarity = 1 / (1 + distance)

            r["similarity"] = similarity

        # Sort (higher similarity = better)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results