
import unittest
import uuid
from unittest.mock import MagicMock

from app.services.chunk_persistence import persist_document_chunks
from app.services.document_chunker import DocumentChunk


class TestChunkPersistence(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.document_id = uuid.uuid4()

    def test_persists_chunks_and_preserves_provenance(self):
        chunks = [
            DocumentChunk(
                chunk_index=0,
                text="First section text",
                page_number=1,
                section_title="Introduction",
            ),
            DocumentChunk(
                chunk_index=1,
                text="Second section text",
                page_number=2,
                section_title="Methods",
            ),
        ]

        result = persist_document_chunks(
            self.db,
            self.document_id,
            chunks,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            [chunk.chunk_index for chunk in result],
            [0, 1],
        )
        self.assertEqual(
            [chunk.text for chunk in result],
            ["First section text", "Second section text"],
        )
        self.assertEqual(
            [chunk.page_number for chunk in result],
            [1, 2],
        )
        self.assertEqual(
            [chunk.section_title for chunk in result],
            ["Introduction", "Methods"],
        )
        self.assertTrue(
            all(chunk.document_id == self.document_id for chunk in result)
        )
        self.assertTrue(
            all(chunk.char_count == len(chunk.text) for chunk in result)
        )

        self.db.add_all.assert_called_once_with(result)
        self.db.flush.assert_called_once_with()
        self.db.commit.assert_not_called()

    def test_empty_chunks_do_not_write_to_database(self):
        result = persist_document_chunks(
            self.db,
            self.document_id,
            [],
        )

        self.assertEqual(result, [])
        self.db.add_all.assert_not_called()
        self.db.flush.assert_not_called()
        self.db.rollback.assert_not_called()

    def test_flush_failure_rolls_back_and_reraises(self):
        self.db.flush.side_effect = RuntimeError("simulated database failure")

        chunks = [
            DocumentChunk(
                chunk_index=0,
                text="Test content",
            )
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "simulated database failure",
        ):
            persist_document_chunks(
                self.db,
                self.document_id,
                chunks,
            )

        self.db.add_all.assert_called_once()
        self.db.flush.assert_called_once()
        self.db.rollback.assert_called_once()
        self.db.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
