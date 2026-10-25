
import unittest
import uuid
from unittest.mock import MagicMock

from app.models.document_chunk_embedding import DocumentChunkEmbedding
from app.services.embedding_persistence import (
    EmbeddingPersistenceError,
    persist_chunk_embeddings,
)
from app.services.embedding_service import EMBEDDING_DIMENSION, MODEL_NAME


class TestEmbeddingPersistence(unittest.TestCase):
    def setUp(self):
        self.db = MagicMock()
        self.chunks = [
            MagicMock(id=uuid.uuid4()),
            MagicMock(id=uuid.uuid4()),
        ]
        self.embeddings = [
            [0.1] * EMBEDDING_DIMENSION,
            [0.2] * EMBEDDING_DIMENSION,
        ]

    def test_persists_embeddings_with_correct_chunk_ids(self):
        result = persist_chunk_embeddings(
            self.db,
            self.chunks,
            self.embeddings,
        )

        self.assertEqual(len(result), 2)
        self.assertTrue(
            all(isinstance(item, DocumentChunkEmbedding) for item in result)
        )
        self.assertEqual(
            [item.chunk_id for item in result],
            [chunk.id for chunk in self.chunks],
        )
        self.assertTrue(
            all(item.model_name == MODEL_NAME for item in result)
        )
        self.assertEqual(
            [len(item.embedding) for item in result],
            [EMBEDDING_DIMENSION, EMBEDDING_DIMENSION],
        )
        self.db.add_all.assert_called_once_with(result)
        self.db.flush.assert_called_once_with()
        self.db.commit.assert_not_called()

    def test_empty_inputs_do_not_write_to_database(self):
        result = persist_chunk_embeddings(self.db, [], [])

        self.assertEqual(result, [])
        self.db.add_all.assert_not_called()
        self.db.flush.assert_not_called()

    def test_mismatched_counts_are_rejected(self):
        with self.assertRaises(EmbeddingPersistenceError):
            persist_chunk_embeddings(
                self.db,
                self.chunks,
                self.embeddings[:1],
            )

        self.db.add_all.assert_not_called()

    def test_wrong_dimensions_are_rejected(self):
        invalid_embeddings = [
            [0.1] * (EMBEDDING_DIMENSION - 1),
            self.embeddings[1],
        ]

        with self.assertRaises(EmbeddingPersistenceError):
            persist_chunk_embeddings(
                self.db,
                self.chunks,
                invalid_embeddings,
            )

        self.db.add_all.assert_not_called()

    def test_non_finite_values_are_rejected(self):
        invalid_embeddings = [
            [float("nan")] + [0.1] * (EMBEDDING_DIMENSION - 1),
            self.embeddings[1],
        ]

        with self.assertRaises(EmbeddingPersistenceError):
            persist_chunk_embeddings(
                self.db,
                self.chunks,
                invalid_embeddings,
            )

        self.db.add_all.assert_not_called()

    def test_flush_failure_rolls_back_and_reraises(self):
        self.db.flush.side_effect = RuntimeError("simulated database failure")

        with self.assertRaisesRegex(
            RuntimeError,
            "simulated database failure",
        ):
            persist_chunk_embeddings(
                self.db,
                self.chunks,
                self.embeddings,
            )

        self.db.add_all.assert_called_once()
        self.db.flush.assert_called_once()
        self.db.rollback.assert_called_once()
        self.db.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
