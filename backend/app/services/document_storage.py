
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings


CHUNK_SIZE = 1024 * 1024


class DocumentUploadError(Exception):
    """Raised when a document cannot be safely stored."""


@dataclass(frozen=True)
class StoredDocument:
    """Metadata describing a successfully stored file."""

    storage_key: str
    checksum_sha256: str
    file_size_bytes: int
    original_filename: str
    content_type: str


def _safe_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    settings = get_settings()

    if extension not in settings.allowed_document_extensions:
        allowed = ", ".join(settings.allowed_document_extensions)
        raise DocumentUploadError(
            f"Unsupported file extension. Allowed extensions: {allowed}"
        )

    return extension


async def store_upload(upload: UploadFile) -> StoredDocument:
    """Stream an upload to private local storage with a size limit."""

    settings = get_settings()

    original_filename = (upload.filename or "").strip()
    if not original_filename:
        raise DocumentUploadError("A filename is required.")

    extension = _safe_extension(original_filename)

    storage_root = settings.storage_dir.resolve()
    storage_root.mkdir(parents=True, exist_ok=True)

    storage_key = f"{uuid.uuid4().hex}{extension}"
    destination = storage_root / storage_key

    checksum = hashlib.sha256()
    total_bytes = 0

    try:
        with destination.open("xb") as output:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > settings.max_upload_size_bytes:
                    raise DocumentUploadError(
                        "File exceeds the maximum upload size of "
                        f"{settings.max_upload_size_bytes} bytes."
                    )

                checksum.update(chunk)
                output.write(chunk)

            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise DocumentUploadError("Empty files are not allowed.")

        return StoredDocument(
            storage_key=storage_key,
            checksum_sha256=checksum.hexdigest(),
            file_size_bytes=total_bytes,
            original_filename=original_filename,
            content_type=(
                upload.content_type or "application/octet-stream"
            ),
        )

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await upload.close()


def delete_stored_document(storage_key: str) -> None:
    """Delete a stored file without allowing path traversal."""

    if (
        not storage_key
        or Path(storage_key).name != storage_key
        or "/" in storage_key
        or "\\" in storage_key
    ):
        raise DocumentUploadError("Invalid storage key.")

    storage_root = get_settings().storage_dir.resolve()
    target = (storage_root / storage_key).resolve()

    if target.parent != storage_root:
        raise DocumentUploadError("Invalid storage path.")

    target.unlink(missing_ok=True)
