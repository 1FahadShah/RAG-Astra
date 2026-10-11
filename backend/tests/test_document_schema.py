
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from pydantic import ValidationError

from app.schemas import DocumentResponse


class TestDocumentResponse(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)

        self.document_data = {
            "id": uuid.uuid4(),
            "filename": "test.pdf",
            "content_type": "application/pdf",
            "file_size_bytes": 1024,
            "checksum_sha256": "a" * 64,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }

    def test_valid_document_is_accepted(self):
        document = DocumentResponse.model_validate(self.document_data)

        self.assertEqual(document.filename, "test.pdf")
        self.assertEqual(document.file_size_bytes, 1024)
        self.assertEqual(len(document.checksum_sha256), 64)

    def test_invalid_checksum_is_rejected(self):
        data = {
            **self.document_data,
            "checksum_sha256": "invalid",
        }

        with self.assertRaises(ValidationError):
            DocumentResponse.model_validate(data)

    def test_negative_file_size_is_rejected(self):
        data = {
            **self.document_data,
            "file_size_bytes": -1,
        }

        with self.assertRaises(ValidationError):
            DocumentResponse.model_validate(data)

    def test_attribute_based_object_is_accepted(self):
        document_object = SimpleNamespace(**self.document_data)

        document = DocumentResponse.model_validate(document_object)

        self.assertEqual(document.id, self.document_data["id"])
        self.assertEqual(document.filename, "test.pdf")


if __name__ == "__main__":
    unittest.main()
