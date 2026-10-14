
import hashlib
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings


CHUNK_SIZE = 1024 * 1024

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ),
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class DocumentUploadError(Exception):
    """Base exception for document upload failures."""

    status_code = 400


class UnsupportedDocumentType(DocumentUploadError):
    """The extension or actual document format is unsupported."""

    status_code = 415


class DocumentTooLarge(DocumentUploadError):
    """The upload exceeds the configured size limit."""

    status_code = 413


class InvalidDocumentContent(DocumentUploadError):
    """The file content does not match its declared format."""

    status_code = 415


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
        raise UnsupportedDocumentType(
            f"Unsupported file extension. Allowed extensions: {allowed}"
        )

    return extension


def _validate_content(path: Path, extension: str) -> None:
    """Validate file content rather than trusting its filename."""

    if extension == ".pdf":
        with path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise InvalidDocumentContent(
                    "The file does not have a valid PDF signature."
                )
        return

    if extension == ".docx":
        try:
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())

                required_entries = {
                    "[Content_Types].xml",
                    "word/document.xml",
                }

                if not required_entries.issubset(names):
                    raise InvalidDocumentContent(
                        "The file is not a valid DOCX document."
                    )

                if archive.testzip() is not None:
                    raise InvalidDocumentContent(
                        "The DOCX archive contains a corrupted file."
                    )

        except zipfile.BadZipFile as exc:
            raise InvalidDocumentContent(
                "The file is not a valid DOCX archive."
            ) from exc

        return

    if extension in {".txt", ".md"}:
        try:
            with path.open("rb") as source:
                while True:
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    if b"\x00" in chunk:
                        raise InvalidDocumentContent(
                            "Text documents must not contain null bytes."
                        )

                    chunk.decode("utf-8")

        except UnicodeDecodeError as exc:
            raise InvalidDocumentContent(
                "Text documents must contain valid UTF-8 text."
            ) from exc


async def store_upload(upload: UploadFile) -> StoredDocument:
    """Stream, validate and store an upload with a size limit."""

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
                    raise DocumentTooLarge(
                        "File exceeds the maximum upload size of "
                        f"{settings.max_upload_size_bytes} bytes."
                    )

                checksum.update(chunk)
                output.write(chunk)

            output.flush()
            os.fsync(output.fileno())

        if total_bytes == 0:
            raise DocumentUploadError("Empty files are not allowed.")

        _validate_content(destination, extension)

        return StoredDocument(
            storage_key=storage_key,
            checksum_sha256=checksum.hexdigest(),
            file_size_bytes=total_bytes,
            original_filename=original_filename,
            content_type=CONTENT_TYPES[extension],
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
