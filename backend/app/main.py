
from fastapi import FastAPI

app = FastAPI(
    title="RAG-Astra API",
    description="Enterprise AI Knowledge & Decision System",
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {
        "status": "ok",
        "service": "rag-astra-api",
    }
