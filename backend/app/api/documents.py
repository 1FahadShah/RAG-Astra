
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.document_storage import (
    DocumentUploadError,
    delete_stored_document,
    store_upload,
)

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


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Store a validated document and persist its metadata."""

    stored = None

    try:
        stored = await store_upload(file)

        # Temporary global duplicate detection. Once knowledge bases
        # and tenant ownership exist, scope this query to the owner
        # and knowledge base.
        existing = (
            db.query(Document)
            .filter(
                Document.checksum_sha256 == stored.checksum_sha256
            )
            .first()
        )

        if existing is not None:
            delete_stored_document(stored.storage_key)
            stored = None
            raise HTTPException(
                status_code=409,
                detail="An identical document has already been uploaded.",
            )

        document = Document(
            filename=stored.original_filename,
            content_type=stored.content_type,
            file_size_bytes=stored.file_size_bytes,
            checksum_sha256=stored.checksum_sha256,
            storage_key=stored.storage_key,
            status="pending",
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return document

    except HTTPException:
        if stored is not None:
            delete_stored_document(stored.storage_key)
        raise

    except DocumentUploadError as exc:
        if stored is not None:
            delete_stored_document(stored.storage_key)

        raise HTTPException(
            status_code=exc.status_code,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        try:
            db.rollback()
        finally:
            if stored is not None:
                delete_stored_document(stored.storage_key)

        raise HTTPException(
            status_code=500,
            detail="Unable to persist document metadata.",
        ) from exc
