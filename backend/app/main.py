
from fastapi import FastAPI

from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Enterprise AI Knowledge & Decision System",
    version=settings.app_version,
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {
        "status": "ok",
        "service": "rag-astra-api",
        "environment": settings.environment,
    }
