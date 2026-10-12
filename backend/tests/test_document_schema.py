
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from pydantic import ValidationError

from app.schemas import (
    DocumentListQuery,
    DocumentListResponse,
    DocumentResponse,
    DocumentStatus,
)


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
        self.assertEqual(document.status, DocumentStatus.PENDING)

    def test_invalid_checksum_is_rejected(self):
        data = {**self.document_data, "checksum_sha256": "invalid"}

        with self.assertRaises(ValidationError):
            DocumentResponse.model_validate(data)

    def test_negative_file_size_is_rejected(self):
        data = {**self.document_data, "file_size_bytes": -1}

        with self.assertRaises(ValidationError):
            DocumentResponse.model_validate(data)

    def test_attribute_based_object_is_accepted(self):
        document_object = SimpleNamespace(**self.document_data)
        document = DocumentResponse.model_validate(document_object)

        self.assertEqual(document.id, self.document_data["id"])
        self.assertEqual(document.filename, "test.pdf")

    def test_unknown_status_is_rejected(self):
        data = {**self.document_data, "status": "unknown"}

        with self.assertRaises(ValidationError):
            DocumentResponse.model_validate(data)

    def test_all_supported_statuses_are_accepted(self):
        for status in DocumentStatus:
            with self.subTest(status=status):
                data = {**self.document_data, "status": status.value}
                document = DocumentResponse.model_validate(data)
                self.assertEqual(document.status, status)


class TestDocumentListQuery(unittest.TestCase):
    def test_default_pagination(self):
        query = DocumentListQuery()

        self.assertEqual(query.limit, 20)
        self.assertEqual(query.offset, 0)

    def test_custom_pagination(self):
        query = DocumentListQuery(limit=50, offset=100)

        self.assertEqual(query.limit, 50)
        self.assertEqual(query.offset, 100)

    def test_limit_above_maximum_is_rejected(self):
        with self.assertRaises(ValidationError):
            DocumentListQuery(limit=101)

    def test_zero_limit_is_rejected(self):
        with self.assertRaises(ValidationError):
            DocumentListQuery(limit=0)

    def test_negative_offset_is_rejected(self):
        with self.assertRaises(ValidationError):
            DocumentListQuery(offset=-1)


class TestDocumentListResponse(unittest.TestCase):
    def test_valid_paginated_response(self):
        now = datetime.now(timezone.utc)
        document_data = {
            "id": uuid.uuid4(),
            "filename": "report.pdf",
            "content_type": "application/pdf",
            "file_size_bytes": 2048,
            "checksum_sha256": "b" * 64,
            "status": "ready",
            "created_at": now,
            "updated_at": now,
        }

        response = DocumentListResponse.model_validate({
            "items": [document_data],
            "total": 101,
            "limit": 20,
            "offset": 40,
        })

        self.assertEqual(len(response.items), 1)
        self.assertEqual(response.total, 101)
        self.assertEqual(response.items[0].status, DocumentStatus.READY)

    def test_negative_total_is_rejected(self):
        with self.assertRaises(ValidationError):
            DocumentListResponse(
                items=[],
                total=-1,
                limit=20,
                offset=0,
            )


if __name__ == "__main__":
    unittest.main()
