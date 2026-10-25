
from uuid import UUID

from sqlalchemy import delete, update
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.chunk_persistence import persist_document_chunks
from app.services.document_chunker import chunk_document
from app.services.document_parser import extract_document


class DocumentProcessingError(Exception):
    """Raised when a document cannot be processed."""


class DocumentNotFoundError(DocumentProcessingError):
    """Raised when the requested document does not exist."""


class InvalidDocumentStateError(DocumentProcessingError):
    """Raised when the document is not eligible for processing."""


def process_document(db: Session, document_id: UUID) -> Document:
    """Atomically claim and process a pending or failed document.

    The processing status is committed before extraction begins.
    Chunk replacement and the final ready status share one transaction.
    """
    claim_result = db.execute(
        update(Document)
        .where(
            Document.id == document_id,
            Document.status.in_(("pending", "failed")),
        )
        .values(status="processing")
        .returning(Document.id)
    )

    claimed_document_id = claim_result.scalar_one_or_none()

    if claimed_document_id is None:
        db.rollback()

        existing_document = db.get(Document, document_id)

        if existing_document is None:
            raise DocumentNotFoundError("Document was not found.")

        raise InvalidDocumentStateError(
            "Cannot process a document with status "
            f"'{existing_document.status}'."
        )

    db.commit()

    document = db.get(Document, document_id)

    if document is None:
        raise DocumentNotFoundError("Document was not found.")

    try:
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

        # Replace old chunks atomically with the new chunks.
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

        # Preserve the failure status in a separate transaction.
        failed_document = db.get(Document, document_id)

        if failed_document is not None:
            failed_document.status = "failed"

            try:
                db.commit()
            except Exception:
                db.rollback()

        raise
