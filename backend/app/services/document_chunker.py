
from dataclasses import dataclass

from app.services.document_parser import ExtractionResult


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of extracted text with its original source provenance."""

    chunk_index: int
    text: str
    page_number: int | None = None
    section_title: str | None = None

    @property
    def char_count(self) -> int:
        """Return the number of Unicode characters in the chunk."""
        return len(self.text)


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into bounded, overlapping character-based chunks."""
    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # Prefer a whitespace boundary in the latter half of the window.
            boundary = max(
                text.rfind(" ", start + chunk_size // 2, end),
                text.rfind("\n", start + chunk_size // 2, end),
            )
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Calculate the next start from the actual end of this chunk.
        next_start = max(start + 1, end - overlap)

        # Move forward to the next word boundary when possible.
        while next_start < end and text[next_start].isspace():
            next_start += 1

        if next_start <= start:
            raise RuntimeError("Chunking failed to make forward progress.")

        start = next_start

    return chunks


def chunk_document(
    extraction: ExtractionResult,
    chunk_size: int = 2000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Create ordered chunks while preserving source provenance.

    Chunk sizing is character-based, not token-based. Tune these initial
    defaults against retrieval evaluation after choosing an embedding model.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[DocumentChunk] = []

    for unit in extraction.units:
        for text in _split_text(unit.text, chunk_size, overlap):
            chunks.append(
                DocumentChunk(
                    chunk_index=len(chunks),
                    text=text,
                    page_number=unit.page_number,
                    section_title=unit.section_title,
                )
            )

    return chunks
