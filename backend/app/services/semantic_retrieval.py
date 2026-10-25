
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_chunk_embedding import DocumentChunkEmbedding
from app.services.embedding_service import MODEL_NAME, embed_query


MAX_RETRIEVAL_LIMIT = 100


class RetrievalInputError(ValueError):
    """Raised when semantic retrieval arguments are invalid."""


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked chunk with its source provenance."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    text: str
    page_number: int | None
    section_title: str | None
    similarity_score: float


def retrieve_chunks(
    db: Session,
    query: str,
    limit: int = 5,
) -> list[RetrievedChunk]:
    """Retrieve ready-document chunks using cosine similarity.

    Only embeddings produced by the configured embedding model are searched.
    Similarity is calculated as 1 minus pgvector cosine distance.
    Higher scores indicate greater similarity; scores are not probabilities.

    This function does not commit, roll back, or modify database records.
    """
    if not isinstance(query, str) or not query.strip():
        raise RetrievalInputError("The search query must be non-empty.")

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_RETRIEVAL_LIMIT
    ):
        raise RetrievalInputError(
            f"Limit must be an integer between 1 and "
            f"{MAX_RETRIEVAL_LIMIT}."
        )

    query_embedding = embed_query(query)

    distance = DocumentChunkEmbedding.embedding.cosine_distance(
        query_embedding
    )

    statement = (
        select(
            DocumentChunk,
            Document,
            distance.label("cosine_distance"),
        )
        .join(
            DocumentChunkEmbedding,
            DocumentChunkEmbedding.chunk_id == DocumentChunk.id,
        )
        .join(
            Document,
            Document.id == DocumentChunk.document_id,
        )
        .where(
            DocumentChunkEmbedding.model_name == MODEL_NAME,
            Document.status == "ready",
        )
        .order_by(distance.asc(), DocumentChunk.id.asc())
        .limit(limit)
    )

    rows = db.execute(statement).all()

    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=document.id,
            filename=document.filename,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            similarity_score=1.0 - float(cosine_distance),
        )
        for chunk, document, cosine_distance in rows
    ]
