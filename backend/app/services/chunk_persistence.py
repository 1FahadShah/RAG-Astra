
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk as StoredDocumentChunk
from app.services.document_chunker import DocumentChunk as Chunk


def persist_document_chunks(
    db: Session,
    document_id: UUID,
    chunks: list[Chunk],
) -> list[StoredDocumentChunk]:
    """Persist all chunks for a document in one database transaction.

    The caller owns the surrounding transaction lifecycle. This function
    flushes changes so database constraint violations are raised, but does
    not commit independently.
    """
    if not chunks:
        return []

    stored_chunks = [
        StoredDocumentChunk(
            document_id=document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            char_count=chunk.char_count,
        )
        for chunk in chunks
    ]

    try:
        db.add_all(stored_chunks)
        db.flush()
    except Exception:
        db.rollback()
        raise

    return stored_chunks
