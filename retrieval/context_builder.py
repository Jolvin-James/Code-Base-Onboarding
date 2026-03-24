from typing import List, Dict


class ContextBuilder:
    """
    Builds LLM-ready context from retrieved chunks.

    Features:
    - Token/character limit enforcement
    - Structured formatting
    - Source tracking
    """

    def __init__(
        self,
        max_context_chars: int = 2500,
        max_chunks: int = 5
    ):
        self.max_context_chars = max_context_chars
        self.max_chunks = max_chunks

    def build_context(self, results: List[Dict]) -> str:
        """
        Convert retrieved chunks into formatted context.
        """

        context_parts = []
        total_chars = 0

        for i, chunk in enumerate(results[:self.max_chunks]):
            formatted_chunk = self._format_chunk(chunk, i)

            chunk_length = len(formatted_chunk)

            # Stop if exceeding limit
            if total_chars + chunk_length > self.max_context_chars:
                break

            context_parts.append(formatted_chunk)
            total_chars += chunk_length

        return "\n\n".join(context_parts)

    def _format_chunk(self, chunk: Dict, idx: int) -> str:
        """
        Format chunk for LLM readability.
        """

        source = chunk.get("source", "unknown")
        header = chunk.get("header", "unknown")

        return (
            f"[Source {idx+1}: {source} | Section: {header}]\n"
            f"{chunk['content']}"
        )