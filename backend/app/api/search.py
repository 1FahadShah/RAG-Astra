
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import (
    HybridSearchRequest,
    HybridSearchResponse,
)
from app.services.hybrid_retrieval import retrieve_hybrid_chunks
from app.services.semantic_retrieval import (
    MAX_RETRIEVAL_LIMIT,
    RetrievalInputError,
)


router = APIRouter(tags=["Search"])


@router.post("/search", response_model=HybridSearchResponse)
def hybrid_search(
    request: HybridSearchRequest,
    db: Session = Depends(get_db),
) -> HybridSearchResponse:
    """Search ready documents using semantic and keyword retrieval."""

    candidate_limit = (
        request.candidate_limit
        if request.candidate_limit is not None
        else min(request.limit * 4, MAX_RETRIEVAL_LIMIT)
    )

    try:
        results = retrieve_hybrid_chunks(
            db=db,
            query=request.query,
            limit=request.limit,
            candidate_limit=request.candidate_limit,
            rrf_k=request.rrf_k,
        )

        return HybridSearchResponse(
            query=request.query,
            items=results,
            limit=request.limit,
            candidate_limit=candidate_limit,
            rrf_k=request.rrf_k,
            returned=len(results),
        )

    except RetrievalInputError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during hybrid search.",
        ) from exc
