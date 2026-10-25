
import unittest
import uuid
from unittest.mock import MagicMock, patch

from app.models.document import Document
from app.services.document_chunker import DocumentChunk
from app.services.document_parser import (
    ExtractionResult,
    ExtractedUnit,
)
from app.services.document_processing import (
    DocumentNotFoundError,
    InvalidDocumentStateError,
    process_document,
)


class TestDocumentProcessing(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.document_id = uuid.uuid4()
        self.document = Document(
            id=self.document_id,
            filename="guide.txt",
            content_type="text/plain",
            file_size_bytes=20,
            checksum_sha256="a" * 64,
            storage_key="stored-guide.txt",
            status="pending",
        )
        self.db.get.return_value = self.document

        self.extraction = ExtractionResult(
            source_filename="guide.txt",
            source_format="txt",
            units=(
                ExtractedUnit(
                    text="RAG-Astra processing test",
                    unit_index=0,
                ),
            ),
        )

        self.chunks = [
            DocumentChunk(
                chunk_index=0,
                text="RAG-Astra processing test",
            )
        ]

    def test_success_marks_document_ready(self):
        with (
            patch(
                "app.services.document_processing.extract_document",
                return_value=self.extraction,
            ) as extract,
            patch(
                "app.services.document_processing.chunk_document",
                return_value=self.chunks,
            ) as chunk,
            patch(
                "app.services.document_processing.persist_document_chunks",
                return_value=[],
            ) as persist,
        ):
            result = process_document(self.db, self.document_id)

        self.assertIs(result, self.document)
        self.assertEqual(result.status, "ready")
        extract.assert_called_once_with(
            "stored-guide.txt",
            "guide.txt",
        )
        chunk.assert_called_once_with(self.extraction)
        persist.assert_called_once_with(
            self.db,
            self.document_id,
            self.chunks,
        )
        self.assertEqual(self.db.commit.call_count, 2)
        self.db.rollback.assert_not_called()

    def test_missing_document_is_rejected(self):
        self.db.get.return_value = None

        with self.assertRaises(DocumentNotFoundError):
            process_document(self.db, self.document_id)

        self.db.commit.assert_not_called()

    def test_non_pending_or_failed_document_is_rejected(self):
        self.document.status = "ready"

        with self.assertRaises(InvalidDocumentStateError):
            process_document(self.db, self.document_id)

        self.db.commit.assert_not_called()

    def test_extraction_failure_marks_document_failed(self):
        with patch(
            "app.services.document_processing.extract_document",
            side_effect=RuntimeError("simulated extraction failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated extraction failure",
            ):
                process_document(self.db, self.document_id)

        self.db.rollback.assert_called_once()
        self.assertEqual(self.document.status, "failed")
        self.assertEqual(self.db.commit.call_count, 2)

    def test_empty_chunks_mark_document_failed(self):
        with (
            patch(
                "app.services.document_processing.extract_document",
                return_value=self.extraction,
            ),
            patch(
                "app.services.document_processing.chunk_document",
                return_value=[],
            ),
            patch(
                "app.services.document_processing.persist_document_chunks",
            ) as persist,
        ):
            with self.assertRaisesRegex(
                Exception,
                "produced no chunks",
            ):
                process_document(self.db, self.document_id)

        persist.assert_not_called()
        self.assertEqual(self.document.status, "failed")


if __name__ == "__main__":
    unittest.main()
