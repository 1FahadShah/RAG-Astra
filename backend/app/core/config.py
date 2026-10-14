
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG-Astra API"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    database_url: str | None = None

    # Upload and storage configuration
    max_upload_size_bytes: int = 10 * 1024 * 1024
    storage_dir: Path = Path("storage")
    allowed_document_extensions: tuple[str, ...] = (
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
