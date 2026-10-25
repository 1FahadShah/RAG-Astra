
from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.core.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Enterprise AI Knowledge & Decision System",
    version=settings.app_version,
)

app.include_router(
    documents_router,
    prefix=settings.api_v1_prefix,
)

app.include_router(
    search_router,
    prefix=settings.api_v1_prefix,
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {
        "status": "ok",
        "service": "rag-astra-api",
        "environment": settings.environment,
    }
