
import unittest
from unittest.mock import patch

from app.services.embedding_service import (
    EMBEDDING_DIMENSION,
    EmbeddingGenerationError,
    EmbeddingInputError,
    embed_documents,
    embed_query,
)


class FakeEmbeddingModel:
    def __init__(self, dimension=EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.received_texts = []
        self.received_options = {}

    def encode(self, texts, **kwargs):
        self.received_texts = list(texts)
        self.received_options = kwargs

        return [
            [0.5] * self.dimension
            for _ in texts
        ]


class TestEmbeddingService(unittest.TestCase):
    def test_embeds_multiple_documents(self):
        model = FakeEmbeddingModel()

        with patch(
            "app.services.embedding_service._get_model",
            return_value=model,
        ):
            vectors = embed_documents(
                ["First document chunk", "Second document chunk"]
            )

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), EMBEDDING_DIMENSION)
        self.assertEqual(len(vectors[1]), EMBEDDING_DIMENSION)
        self.assertEqual(
            model.received_texts,
            ["First document chunk", "Second document chunk"],
        )
        self.assertTrue(
            model.received_options["normalize_embeddings"]
        )

    def test_query_uses_retrieval_instruction(self):
        model = FakeEmbeddingModel()

        with patch(
            "app.services.embedding_service._get_model",
            return_value=model,
        ):
            vector = embed_query("How does retrieval work?")

        self.assertEqual(len(vector), EMBEDDING_DIMENSION)
        self.assertTrue(
            model.received_texts[0].startswith(
                "Represent this sentence for searching relevant passages: "
            )
        )

    def test_empty_document_list_returns_empty_list(self):
        with patch(
            "app.services.embedding_service._get_model"
        ) as get_model:
            self.assertEqual(embed_documents([]), [])
            get_model.assert_not_called()

    def test_rejects_blank_document_text(self):
        with self.assertRaises(EmbeddingInputError):
            embed_documents(["Valid text", "   "])

    def test_rejects_blank_query(self):
        with self.assertRaises(EmbeddingInputError):
            embed_query("   ")

    def test_rejects_wrong_embedding_dimension(self):
        model = FakeEmbeddingModel(dimension=10)

        with patch(
            "app.services.embedding_service._get_model",
            return_value=model,
        ):
            with self.assertRaises(EmbeddingGenerationError):
                embed_documents(["Test document"])


if __name__ == "__main__":
    unittest.main()
