
import math

from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk
from app.models.document_chunk_embedding import DocumentChunkEmbedding
from app.services.embedding_service import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
)


class EmbeddingPersistenceError(ValueError):
    """Raised when embedding data cannot be persisted safely."""


def persist_chunk_embeddings(
    db: Session,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
    model_name: str = MODEL_NAME,
) -> list[DocumentChunkEmbedding]:
    """Persist embeddings for existing chunks without committing.

    The caller owns the surrounding transaction. Each vector is
    associated with the corresponding persisted chunk by its ID.
    """
    if len(chunks) != len(embeddings):
        raise EmbeddingPersistenceError(
            "The number of embeddings must match the number of chunks."
        )

    if not isinstance(model_name, str) or not model_name.strip():
        raise EmbeddingPersistenceError(
            "The embedding model name must be non-empty."
        )

    if not chunks:
        return []

    records: list[DocumentChunkEmbedding] = []

    for chunk, vector in zip(chunks, embeddings, strict=True):
        if chunk.id is None:
            raise EmbeddingPersistenceError(
                "Every chunk must be persisted before its embedding."
            )

        if len(vector) != EMBEDDING_DIMENSION:
            raise EmbeddingPersistenceError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings."
            )

        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in vector
        ):
            raise EmbeddingPersistenceError(
                "Embedding values must be finite numbers."
            )

        records.append(
            DocumentChunkEmbedding(
                chunk_id=chunk.id,
                model_name=model_name,
                embedding=[float(value) for value in vector],
            )
        )

    try:
        db.add_all(records)
        db.flush()
    except Exception:
        db.rollback()
        raise

    return records
