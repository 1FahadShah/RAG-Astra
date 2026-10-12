
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document
from app.schemas.document import DocumentListResponse

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """Return a paginated list of document metadata."""

    total = db.query(Document).count()

    documents = (
        db.query(Document)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return DocumentListResponse(
        items=documents,
        total=total,
        limit=limit,
        offset=offset,
    )
