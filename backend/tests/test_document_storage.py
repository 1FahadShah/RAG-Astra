
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.services.document_storage import (
    DocumentUploadError,
    InvalidDocumentContent,
    delete_stored_document,
    store_upload,
)


class TestDocumentStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name)

        self.test_settings = SimpleNamespace(
            storage_dir=self.storage_path,
            max_upload_size_bytes=16,
            allowed_document_extensions=(
                ".pdf",
                ".docx",
                ".txt",
                ".md",
            ),
        )

        self.settings_patcher = patch(
            "app.services.document_storage.get_settings",
            return_value=self.test_settings,
        )
        self.settings_patcher.start()

    def tearDown(self):
        self.settings_patcher.stop()
        self.temp_dir.cleanup()

    def make_upload(self, filename, content):
        return UploadFile(
            filename=filename,
            file=io.BytesIO(content),
            headers=Headers({"content-type": "application/pdf"}),
        )

    async def test_valid_upload_is_stored_with_checksum(self):
        content = b"%PDF-1.4\n"

        result = await store_upload(
            self.make_upload("report.pdf", content)
        )

        stored_path = self.storage_path / result.storage_key

        self.assertTrue(stored_path.is_file())
        self.assertEqual(stored_path.read_bytes(), content)
        self.assertEqual(result.file_size_bytes, len(content))
        self.assertEqual(
            result.checksum_sha256,
            hashlib.sha256(content).hexdigest(),
        )
        self.assertEqual(result.content_type, "application/pdf")
        self.assertEqual(
            Path(result.storage_key).name,
            result.storage_key,
        )

    async def test_invalid_pdf_signature_is_rejected(self):
        with self.assertRaises(InvalidDocumentContent):
            await store_upload(
                self.make_upload("fake.pdf", b"not a PDF")
            )

        self.assertEqual(list(self.storage_path.iterdir()), [])

    async def test_empty_upload_is_rejected(self):
        with self.assertRaises(DocumentUploadError):
            await store_upload(
                self.make_upload("empty.pdf", b"")
            )

        self.assertEqual(list(self.storage_path.iterdir()), [])

    async def test_unsupported_extension_is_rejected(self):
        with self.assertRaises(DocumentUploadError):
            await store_upload(
                self.make_upload("script.exe", b"content")
            )

        self.assertEqual(list(self.storage_path.iterdir()), [])

    async def test_oversized_upload_is_rejected_and_cleaned_up(self):
        with self.assertRaises(DocumentUploadError):
            await store_upload(
                self.make_upload("large.txt", b"x" * 17)
            )

        self.assertEqual(list(self.storage_path.iterdir()), [])

    async def test_delete_rejects_path_traversal(self):
        with self.assertRaises(DocumentUploadError):
            delete_stored_document("../outside.txt")

    async def test_delete_removes_stored_file(self):
        result = await store_upload(
            self.make_upload("report.txt", b"small")
        )
        stored_path = self.storage_path / result.storage_key

        delete_stored_document(result.storage_key)

        self.assertFalse(stored_path.exists())


if __name__ == "__main__":
    unittest.main()
