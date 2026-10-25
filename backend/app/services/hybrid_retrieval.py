
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.services.keyword_retrieval import retrieve_keyword_chunks
from app.services.semantic_retrieval import (
    MAX_RETRIEVAL_LIMIT,
    RetrievalInputError,
    retrieve_chunks,
)


DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class HybridRetrievedChunk:
    """A fused retrieval result with source provenance and ranking signals."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    text: str
    page_number: int | None
    section_title: str | None
    semantic_rank: int | None
    keyword_rank: int | None
    similarity_score: float | None
    keyword_score: float | None
    rrf_score: float


def retrieve_hybrid_chunks(
    db: Session,
    query: str,
    limit: int = 5,
    candidate_limit: int | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridRetrievedChunk]:
    """Combine semantic and keyword rankings using Reciprocal Rank Fusion.

    Both retrieval services are called independently. RRF combines their
    rank positions rather than comparing their raw scores.

    A chunk returned by both services is included only once. Its contribution
    from each ranking is added to its final RRF score.

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

    if (
        not isinstance(rrf_k, int)
        or isinstance(rrf_k, bool)
        or rrf_k < 1
    ):
        raise RetrievalInputError("RRF k must be a positive integer.")

    if candidate_limit is None:
        candidate_limit = min(limit * 4, MAX_RETRIEVAL_LIMIT)
    elif (
        not isinstance(candidate_limit, int)
        or isinstance(candidate_limit, bool)
        or not 1 <= candidate_limit <= MAX_RETRIEVAL_LIMIT
    ):
        raise RetrievalInputError(
            f"Candidate limit must be an integer between 1 and "
            f"{MAX_RETRIEVAL_LIMIT}."
        )

    semantic_results = retrieve_chunks(
        db,
        query.strip(),
        limit=candidate_limit,
    )
    keyword_results = retrieve_keyword_chunks(
        db,
        query.strip(),
        limit=candidate_limit,
    )

    # Each chunk is keyed by its unique ID, so overlap is fused, not duplicated.
    fused: dict[UUID, dict] = {}

    for rank, result in enumerate(semantic_results, start=1):
        entry = fused.setdefault(
            result.chunk_id,
            {
                "result": result,
                "semantic_rank": None,
                "keyword_rank": None,
                "similarity_score": None,
                "keyword_score": None,
                "rrf_score": 0.0,
            },
        )

        entry["semantic_rank"] = rank
        entry["similarity_score"] = float(result.similarity_score)
        entry["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, result in enumerate(keyword_results, start=1):
        entry = fused.setdefault(
            result.chunk_id,
            {
                "result": result,
                "semantic_rank": None,
                "keyword_rank": None,
                "similarity_score": None,
                "keyword_score": None,
                "rrf_score": 0.0,
            },
        )

        entry["keyword_rank"] = rank
        entry["keyword_score"] = float(result.keyword_score)
        entry["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked_entries = sorted(
        fused.values(),
        key=lambda entry: (
            -entry["rrf_score"],
            str(entry["result"].chunk_id),
        ),
    )

    return [
        HybridRetrievedChunk(
            chunk_id=entry["result"].chunk_id,
            document_id=entry["result"].document_id,
            filename=entry["result"].filename,
            chunk_index=entry["result"].chunk_index,
            text=entry["result"].text,
            page_number=entry["result"].page_number,
            section_title=entry["result"].section_title,
            semantic_rank=entry["semantic_rank"],
            keyword_rank=entry["keyword_rank"],
            similarity_score=entry["similarity_score"],
            keyword_score=entry["keyword_score"],
            rrf_score=entry["rrf_score"],
        )
        for entry in ranked_entries[:limit]
    ]
