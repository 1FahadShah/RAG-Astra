
import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app
from app.models.document import Document


class TestDocumentListAPI(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.document = Document(
            id=uuid.uuid4(),
            filename="report.pdf",
            content_type="application/pdf",
            file_size_bytes=2048,
            checksum_sha256="a" * 64,
            status="ready",
            created_at=self.now,
            updated_at=self.now,
        )

        self.db = MagicMock()
        query = self.db.query.return_value
        query.count.return_value = 1
        query.order_by.return_value = query
        query.offset.return_value = query
        query.limit.return_value = query
        query.all.return_value = [self.document]

        def override_get_db():
            yield self.db

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)

    def test_list_documents_returns_paginated_response(self):
        response = self.client.get("/api/v1/documents")

        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["total"], 1)
        self.assertEqual(data["limit"], 20)
        self.assertEqual(data["offset"], 0)
        self.assertEqual(len(data["items"]), 1)
        self.assertEqual(data["items"][0]["filename"], "report.pdf")
        self.assertEqual(data["items"][0]["status"], "ready")

    def test_custom_pagination_is_applied(self):
        response = self.client.get(
            "/api/v1/documents",
            params={"limit": 10, "offset": 20},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["limit"], 10)
        self.assertEqual(response.json()["offset"], 20)

        self.db.query.return_value.offset.assert_called_once_with(20)
        self.db.query.return_value.limit.assert_called_once_with(10)

    def test_limit_above_maximum_is_rejected(self):
        response = self.client.get(
            "/api/v1/documents",
            params={"limit": 101},
        )

        self.assertEqual(response.status_code, 422)

    def test_zero_limit_is_rejected(self):
        response = self.client.get(
            "/api/v1/documents",
            params={"limit": 0},
        )

        self.assertEqual(response.status_code, 422)

    def test_negative_offset_is_rejected(self):
        response = self.client.get(
            "/api/v1/documents",
            params={"offset": -1},
        )

        self.assertEqual(response.status_code, 422)

    def test_empty_document_list(self):
        self.db.query.return_value.count.return_value = 0
        self.db.query.return_value.all.return_value = []

        response = self.client.get("/api/v1/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"items": [], "total": 0, "limit": 20, "offset": 0},
        )


if __name__ == "__main__":
    unittest.main()
