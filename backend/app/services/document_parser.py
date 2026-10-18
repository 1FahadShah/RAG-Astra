
from dataclasses import dataclass
from pathlib import Path
import re

import pymupdf
from docx import Document as WordDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.config import get_settings


class DocumentParsingError(Exception):
    """Base exception for document text extraction failures."""


class UnsupportedDocumentFormat(DocumentParsingError):
    """The stored document format is not supported."""


class InvalidDocumentForParsing(DocumentParsingError):
    """The stored document is corrupt or cannot be parsed."""


class EmptyExtractedText(DocumentParsingError):
    """The document contains no usable extractable text."""


class ScannedPdfError(EmptyExtractedText):
    """The PDF has no extractable text and may require OCR."""


@dataclass(frozen=True)
class ExtractedUnit:
    """A text unit with source provenance where available."""

    text: str
    unit_index: int
    page_number: int | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    """Extracted text and metadata for a stored document."""

    source_filename: str
    source_format: str
    units: tuple[ExtractedUnit, ...]

    @property
    def full_text(self) -> str:
        """Return all extracted units as one readable text string."""
        separator = (
            "\n"
            if self.source_format in {"txt", "md"}
            else "\n\n"
        )
        return separator.join(unit.text for unit in self.units)

def _normalise_text(text: str) -> str:
    """Normalise line endings and remove unnecessary whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    normalised_lines = []

    for line in lines:
        if line:
            normalised_lines.append(line)

    return "\n".join(normalised_lines).strip()


def _resolve_stored_path(storage_key: str) -> Path:
    """Resolve a storage key without allowing path traversal."""
    if (
        not storage_key
        or Path(storage_key).name != storage_key
        or "/" in storage_key
        or "\\" in storage_key
    ):
        raise DocumentParsingError("Invalid document storage key.")

    storage_root = get_settings().storage_dir.resolve()
    path = (storage_root / storage_key).resolve()

    if path.parent != storage_root:
        raise DocumentParsingError("Invalid document storage path.")

    if not path.is_file():
        raise DocumentParsingError("Stored document file was not found.")

    return path


def _extract_pdf(path: Path) -> tuple[ExtractedUnit, ...]:
    """Extract non-empty text from each PDF page."""
    units: list[ExtractedUnit] = []

    try:
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = _normalise_text(page.get_text("text"))

                if text:
                    units.append(
                        ExtractedUnit(
                            text=text,
                            unit_index=len(units),
                            page_number=page_number,
                        )
                    )
    except Exception as exc:
        raise InvalidDocumentForParsing(
            "The PDF could not be opened or parsed."
        ) from exc

    if not units:
        raise ScannedPdfError(
            "No extractable text was found in this PDF. "
            "It may be scanned or image-only; OCR is not enabled."
        )

    return tuple(units)


def _extract_docx(path: Path) -> tuple[ExtractedUnit, ...]:
    """Extract Word paragraphs and tables in document order."""
    units: list[ExtractedUnit] = []
    current_section: str | None = None

    try:
        document = WordDocument(path)

        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text = _normalise_text(item.text)
                if not text:
                    continue

                style_name = item.style.name if item.style else ""
                if style_name.startswith("Heading"):
                    current_section = text

                units.append(
                    ExtractedUnit(
                        text=text,
                        unit_index=len(units),
                        section_title=current_section,
                    )
                )

            elif isinstance(item, Table):
                for row in item.rows:
                    cells = [
                        _normalise_text(cell.text)
                        for cell in row.cells
                    ]
                    cells = [cell for cell in cells if cell]

                    if cells:
                        units.append(
                            ExtractedUnit(
                                text=" | ".join(cells),
                                unit_index=len(units),
                                section_title=current_section,
                            )
                        )
    except Exception as exc:
        raise InvalidDocumentForParsing(
            "The DOCX document could not be opened or parsed."
        ) from exc

    if not units:
        raise EmptyExtractedText(
            "No extractable text was found in this DOCX document."
        )

    return tuple(units)


def _extract_plain_text(path: Path) -> tuple[ExtractedUnit, ...]:
    """Read UTF-8 text without discarding Markdown syntax."""
    try:
        text = _normalise_text(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, OSError) as exc:
        raise InvalidDocumentForParsing(
            "The text document could not be read as UTF-8."
        ) from exc

    if not text:
        raise EmptyExtractedText(
            "No extractable text was found in this document."
        )

    return (ExtractedUnit(text=text, unit_index=0),)


def extract_document(
    storage_key: str,
    source_filename: str | None = None,
) -> ExtractionResult:
    """Extract text from a stored PDF, DOCX, TXT, or Markdown file."""
    path = _resolve_stored_path(storage_key)
    extension = path.suffix.lower()

    if extension == ".pdf":
        units = _extract_pdf(path)
    elif extension == ".docx":
        units = _extract_docx(path)
    elif extension in {".txt", ".md"}:
        units = _extract_plain_text(path)
    else:
        raise UnsupportedDocumentFormat(
            f"Unsupported document format: {extension or '(no extension)'}."
        )

    return ExtractionResult(
        source_filename=source_filename or storage_key,
        source_format=extension.lstrip("."),
        units=units,
    )
