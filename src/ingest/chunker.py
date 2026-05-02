"""Semantic chunking with deduplication and rich metadata."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_PAGE_RE = re.compile(r"\[Page (\d+)\]")


@dataclass
class Chunk:
    """A single text chunk with provenance metadata."""

    id: str
    text: str
    source: str
    doc_type: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _detect_page(text: str, preceding_text: str) -> int | None:
    """Find the most recent [Page N] marker before this chunk's position."""
    matches = list(_PAGE_RE.finditer(preceding_text))
    if matches:
        return int(matches[-1].group(1))
    return None


class SemanticChunker:
    """Split documents into overlapping chunks and deduplicate by content hash."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        text: str,
        source: str,
        doc_type: str,
    ) -> list[Chunk]:
        """Split *text* into chunks and attach source metadata.

        Deduplicates by SHA-256 of the stripped content so re-ingesting the
        same document produces identical chunk IDs.
        """
        raw_chunks = self._splitter.split_text(text)
        seen_hashes: set[str] = set()
        chunks: list[Chunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            stripped = chunk_text.strip()
            if not stripped:
                continue

            content_hash = _content_hash(stripped)
            if content_hash in seen_hashes:
                logger.debug("Skipping duplicate chunk at index %d (hash=%s)", idx, content_hash)
                continue
            seen_hashes.add(content_hash)

            # Reconstruct the position in the original text to find page numbers
            position = text.find(chunk_text)
            preceding = text[:position] if position >= 0 else ""
            page_number = _detect_page(chunk_text, preceding) if doc_type == "pdf" else None

            chunk_id = f"{_content_hash(source)}-{content_hash}"

            metadata: dict[str, Any] = {
                "source": source,
                "doc_type": doc_type,
                "chunk_index": idx,
                "char_count": len(stripped),
                "content_hash": content_hash,
            }
            if page_number is not None:
                metadata["page_number"] = page_number

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=stripped,
                    source=source,
                    doc_type=doc_type,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )

        logger.info(
            "Chunked %s: %d raw → %d unique chunks (size=%d, overlap=%d)",
            source,
            len(raw_chunks),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
