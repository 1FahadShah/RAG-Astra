
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class HybridSearchRequest(BaseModel):
    """Validated request for hybrid knowledge retrieval."""

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=100)
    candidate_limit: int | None = Field(default=None, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=10000)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Trim surrounding whitespace and reject blank queries."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("The search query must be non-empty.")
        return normalized


class HybridSearchResult(BaseModel):
    """Public representation of a fused retrieval result."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int = Field(ge=0)
    text: str
    page_number: int | None = Field(default=None, ge=1)
    section_title: str | None
    semantic_rank: int | None = Field(default=None, ge=1)
    keyword_rank: int | None = Field(default=None, ge=1)
    similarity_score: float | None
    keyword_score: float | None = Field(default=None, ge=0)
    rrf_score: float = Field(ge=0)


class HybridSearchResponse(BaseModel):
    """Hybrid search results and the effective retrieval settings."""

    query: str
    items: list[HybridSearchResult]
    limit: int = Field(ge=1, le=100)
    candidate_limit: int = Field(ge=1, le=100)
    rrf_k: int = Field(ge=1, le=10000)
    returned: int = Field(ge=0)
