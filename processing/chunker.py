# processing/chunker.py

import re
from typing import List, Dict, Any


class MarkdownChunker:
    """
    Production-ready Markdown Header Chunker.

    Features:
    - Header-based semantic chunking (#, ##, ###, ...)
    - Code block preservation
    - Text preprocessing
    - Small chunk merging
    - Metadata preservation
    """

    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.*)')
    CODE_BLOCK_PATTERN = re.compile(r'```.*?```', re.DOTALL)

    def __init__(self, min_chunk_size: int = 100):
        self.min_chunk_size = min_chunk_size

    # -----------------------------
    # PUBLIC METHOD
    # -----------------------------
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main entry point.

        Input:
            documents = [
                {
                    "content": "...",
                    "source": "...",
                    ...
                }
            ]

        Output:
            List of chunk dictionaries
        """

        all_chunks = []

        for doc in documents:
            content = self._preprocess_text(doc["content"])

            # Extract code blocks
            content, code_blocks = self._extract_code_blocks(content)

            chunks = self._split_into_chunks(content)

            # Restore code blocks
            chunks = [self._restore_code_blocks(chunk, code_blocks) for chunk in chunks]

            # Attach metadata
            enriched_chunks = self._attach_metadata(chunks, doc)

            # Merge small chunks
            merged_chunks = self._merge_small_chunks(enriched_chunks)

            all_chunks.extend(merged_chunks)

        return all_chunks

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    def _preprocess_text(self, text: str) -> str:
        """
        Normalize markdown text.
        """
        text = text.replace("\r\n", "\n")

        # Remove excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    # -----------------------------
    # CODE BLOCK HANDLING
    # -----------------------------
    def _extract_code_blocks(self, text: str):
        """
        Replace code blocks with placeholders.
        """
        code_blocks = []

        def replacer(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        text = re.sub(self.CODE_BLOCK_PATTERN, replacer, text)
        return text, code_blocks

    def _restore_code_blocks(self, chunk: Dict, code_blocks: List[str]):
        """
        Restore code blocks inside chunk content.
        """
        content = chunk["content"]

        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            content = content.replace(placeholder, block)

        chunk["content"] = content
        return chunk

    # -----------------------------
    # CORE CHUNKING
    # -----------------------------
    def _split_into_chunks(self, text: str) -> List[Dict]:
        """
        Split markdown text based on headers.
        """
        lines = text.split("\n")

        chunks = []
        current_chunk = []
        current_header = "ROOT"
        current_level = 0

        for line in lines:
            header_match = self.HEADER_PATTERN.match(line)

            if header_match:
                # Save previous chunk
                if current_chunk:
                    chunks.append({
                        "header": current_header,
                        "level": current_level,
                        "content": "\n".join(current_chunk).strip()
                    })

                # Start new chunk
                current_header = header_match.group(2).strip()
                current_level = len(header_match.group(1))
                current_chunk = []

            else:
                current_chunk.append(line)

        # Final chunk
        if current_chunk:
            chunks.append({
                "header": current_header,
                "level": current_level,
                "content": "\n".join(current_chunk).strip()
            })

        return chunks

    # -----------------------------
    # METADATA ATTACHMENT
    # -----------------------------
    def _attach_metadata(self, chunks: List[Dict], doc: Dict) -> List[Dict]:
        """
        Attach document-level metadata to each chunk.
        """
        enriched = []

        for chunk in chunks:
            enriched.append({
                "content": chunk["content"],
                "header": chunk["header"],
                "level": chunk["level"],
                "source": doc["source"],
                "last_updated": doc.get("last_updated"),
                "last_updated_readable": doc.get("last_updated_readable"),
            })

        return enriched

    # -----------------------------
    # MERGE SMALL CHUNKS
    # -----------------------------
    def _merge_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Merge chunks that are too small.
        """
        if not chunks:
            return []

        merged = []
        buffer = None

        for chunk in chunks:
            if len(chunk["content"]) < self.min_chunk_size:
                if buffer:
                    buffer["content"] += "\n" + chunk["content"]
                else:
                    buffer = chunk.copy()
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = None
                merged.append(chunk)

        if buffer:
            merged.append(buffer)

        return merged