from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, literal_column, select
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.semantic_retrieval import (
    MAX_RETRIEVAL_LIMIT,
    RetrievalInputError,
)


@dataclass(frozen=True)
class KeywordRetrievedChunk:
    """A keyword-matched chunk with source provenance."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    text: str
    page_number: int | None
    section_title: str | None
    keyword_score: float


def retrieve_keyword_chunks(
    db: Session,
    query: str,
    limit: int = 5,
) -> list[KeywordRetrievedChunk]:
    """Retrieve ready-document chunks using PostgreSQL full-text search.

    Uses the English text-search configuration and the matching GIN index.
    Higher ts_rank_cd scores indicate stronger lexical relevance; they are
    ranking signals, not probabilities.

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

    # Keep the configuration literal consistent with the expression index.
    english_config = literal_column("'english'")

    search_vector = func.to_tsvector(
        english_config,
        DocumentChunk.text,
    )
    query_vector = func.plainto_tsquery(
        english_config,
        query.strip(),
    )
    rank = func.ts_rank_cd(search_vector, query_vector)

    statement = (
        select(
            DocumentChunk,
            Document,
            rank.label("keyword_score"),
        )
        .join(
            Document,
            Document.id == DocumentChunk.document_id,
        )
        .where(
            Document.status == "ready",
            search_vector.op("@@")(query_vector),
        )
        .order_by(rank.desc(), DocumentChunk.id.asc())
        .limit(limit)
    )

    rows = db.execute(statement).all()

    return [
        KeywordRetrievedChunk(
            chunk_id=chunk.id,
            document_id=document.id,
            filename=document.filename,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            keyword_score=float(keyword_score),
        )
        for chunk, document, keyword_score in rows
    ]