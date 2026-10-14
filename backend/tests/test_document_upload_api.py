
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.models.document import Document


class TestDocumentUploadAPI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)

        self.settings = SimpleNamespace(
            storage_dir=self.storage_path,
            max_upload_size_bytes=1024,
            allowed_document_extensions=(
                ".pdf",
                ".docx",
                ".txt",
                ".md",
            ),
        )

        self.settings_patcher = patch(
            "app.services.document_storage.get_settings",
            return_value=self.settings,
        )
        self.settings_patcher.start()

        self.db = MagicMock()
        self.db.query.return_value.filter.return_value.first.return_value = (
            None
        )
        self.db.refresh.side_effect = self.populate_generated_fields

        def override_get_db():
            yield self.db

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)
        self.settings_patcher.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def populate_generated_fields(document):
        now = datetime.now(timezone.utc)

        if document.id is None:
            document.id = uuid.uuid4()

        if document.created_at is None:
            document.created_at = now

        if document.updated_at is None:
            document.updated_at = now

    def upload_pdf(self, content=b"%PDF-1.4\n"):
        return self.client.post(
            "/api/v1/documents",
            files={
                "file": (
                    "report.pdf",
                    content,
                    "application/pdf",
                )
            },
        )

    def test_valid_upload_persists_metadata(self):
        response = self.upload_pdf()

        self.assertEqual(response.status_code, 201)

        data = response.json()
        self.assertEqual(data["filename"], "report.pdf")
        self.assertEqual(data["content_type"], "application/pdf")
        self.assertEqual(data["file_size_bytes"], len(b"%PDF-1.4\n"))
        self.assertEqual(data["status"], "pending")
        self.assertEqual(len(data["checksum_sha256"]), 64)

        # Internal storage paths must not be exposed by the API.
        self.assertNotIn("storage_key", data)

        self.db.add.assert_called_once()
        self.db.commit.assert_called_once()
        self.db.refresh.assert_called_once()

        stored_document = self.db.add.call_args.args[0]
        stored_path = self.storage_path / stored_document.storage_key

        self.assertTrue(stored_path.is_file())
        self.assertEqual(stored_path.read_bytes(), b"%PDF-1.4\n")

    def test_duplicate_upload_returns_conflict_and_removes_new_file(self):
        self.db.query.return_value.filter.return_value.first.return_value = (
            object()
        )

        response = self.upload_pdf()

        self.assertEqual(response.status_code, 409)
        self.db.add.assert_not_called()
        self.db.commit.assert_not_called()
        self.assertEqual(list(self.storage_path.iterdir()), [])

    def test_invalid_pdf_returns_415(self):
        response = self.upload_pdf(b"not a PDF")

        self.assertEqual(response.status_code, 415)
        self.db.add.assert_not_called()
        self.assertEqual(list(self.storage_path.iterdir()), [])

    def test_database_failure_rolls_back_and_removes_file(self):
        self.db.commit.side_effect = RuntimeError("simulated failure")

        response = self.upload_pdf()

        self.assertEqual(response.status_code, 500)
        self.db.rollback.assert_called_once()
        self.assertEqual(list(self.storage_path.iterdir()), [])

    def test_unsupported_extension_is_rejected(self):
        response = self.client.post(
            "/api/v1/documents",
            files={"file": ("script.exe", b"content", "text/plain")},
        )

        self.assertEqual(response.status_code, 415)
        self.db.add.assert_not_called()
        self.assertEqual(list(self.storage_path.iterdir()), [])

    def test_oversized_upload_is_rejected(self):
        response = self.upload_pdf(b"%PDF-" + b"x" * 1024)

        self.assertEqual(response.status_code, 413)
        self.db.add.assert_not_called()
        self.assertEqual(list(self.storage_path.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
