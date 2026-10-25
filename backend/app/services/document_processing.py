
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.document import Document
from app.services.chunk_persistence import persist_document_chunks
from app.services.document_chunker import chunk_document
from app.services.document_parser import extract_document
from sqlalchemy import delete

from app.models.document_chunk import DocumentChunk

class DocumentProcessingError(Exception):
    """Raised when a document cannot be processed."""


class DocumentNotFoundError(DocumentProcessingError):
    """Raised when the requested document does not exist."""


class InvalidDocumentStateError(DocumentProcessingError):
    """Raised when the document is not eligible for processing."""


def process_document(db: Session, document_id: UUID) -> Document:
    """Extract, chunk, and persist a pending document.

    The processing status is committed first. Chunks and the final ready
    status are committed together in a second transaction.
    """
    document = db.get(Document, document_id)

    if document is None:
        raise DocumentNotFoundError("Document was not found.")

    if document.status not in {"pending", "failed"}:
        raise InvalidDocumentStateError(
            f"Cannot process a document with status '{document.status}'."
        )

    try:
        document.status = "processing"
        db.commit()

        if not document.storage_key:
            raise DocumentProcessingError(
                "The document has no stored file."
            )

        extraction = extract_document(
            document.storage_key,
            document.filename,
        )
        chunks = chunk_document(extraction)

        if not chunks:
            raise DocumentProcessingError(
                "Document processing produced no chunks."
            )

        db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document.id
            )
        )

        persist_document_chunks(db, document.id, chunks)

        document.status = "ready"
        db.commit()
        return document

    except Exception:
        db.rollback()

        # Record failure in a separate transaction so the failed status
        # survives rollback of chunk persistence.
        failed_document = db.get(Document, document_id)

        if failed_document is not None:
            failed_document.status = "failed"
            try:
                db.commit()
            except Exception:
                db.rollback()

        raise
