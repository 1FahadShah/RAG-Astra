
import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.services.document_parser import DocumentParsingError
from app.services.document_processing import (
    DocumentNotFoundError,
    DocumentProcessingError,
    InvalidDocumentStateError,
)


class TestDocumentProcessingAPI(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.document_id = uuid.uuid4()

        now = datetime.now(timezone.utc)
        self.document = MagicMock()
        self.document.id = self.document_id
        self.document.filename = "guide.txt"
        self.document.content_type = "text/plain"
        self.document.file_size_bytes = 20
        self.document.checksum_sha256 = "a" * 64
        self.document.status = "ready"
        self.document.created_at = now
        self.document.updated_at = now

        def override_get_db():
            yield self.db

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

        self.processor = patch(
            "app.api.documents.process_document"
        )
        self.process_mock = self.processor.start()
        self.addCleanup(self.processor.stop)
        self.addCleanup(
            lambda: app.dependency_overrides.pop(get_db, None)
        )

    def process_url(self):
        return (
            f"/api/v1/documents/{self.document_id}/process"
        )

    def test_success_returns_updated_document(self):
        self.process_mock.return_value = self.document

        response = self.client.post(self.process_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.process_mock.assert_called_once_with(
            self.db,
            self.document_id,
        )

    def test_missing_document_returns_404(self):
        self.process_mock.side_effect = DocumentNotFoundError(
            "Document was not found."
        )

        response = self.client.post(self.process_url())

        self.assertEqual(response.status_code, 404)

    def test_invalid_document_state_returns_409(self):
        self.process_mock.side_effect = InvalidDocumentStateError(
            "Document is already processing."
        )

        response = self.client.post(self.process_url())

        self.assertEqual(response.status_code, 409)

    def test_processing_error_returns_422(self):
        self.process_mock.side_effect = DocumentProcessingError(
            "No chunks were produced."
        )

        response = self.client.post(self.process_url())

        self.assertEqual(response.status_code, 422)
        self.assertNotIn("No chunks were produced", response.text)

    def test_parsing_error_returns_422(self):
        self.process_mock.side_effect = DocumentParsingError(
            "Internal parser detail"
        )

        response = self.client.post(self.process_url())

        self.assertEqual(response.status_code, 422)
        self.assertNotIn("Internal parser detail", response.text)

    def test_unexpected_error_returns_safe_500(self):
        self.process_mock.side_effect = RuntimeError(
            "Internal database detail"
        )

        response = self.client.post(self.process_url())

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("Internal database detail", response.text)


if __name__ == "__main__":
    unittest.main()
