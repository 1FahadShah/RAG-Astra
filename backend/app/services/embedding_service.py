
from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)


class EmbeddingInputError(ValueError):
    """Raised when embedding input is invalid."""


class EmbeddingGenerationError(RuntimeError):
    """Raised when generated embeddings have an invalid shape."""


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the embedding model once per process."""
    return SentenceTransformer(MODEL_NAME)


def _encode_texts(
    texts: Sequence[str],
    *,
    is_query: bool,
) -> list[list[float]]:
    """Generate normalized, 384-dimensional embeddings."""
    if isinstance(texts, str):
        raise EmbeddingInputError(
            "Expected a sequence of texts, not a single string."
        )

    if not texts:
        return []

    if any(
        not isinstance(item, str) or not item.strip()
        for item in texts
    ):
        raise EmbeddingInputError(
            "Every input text must be a non-empty string."
        )

    model_inputs = [
        f"{QUERY_INSTRUCTION}{item}" if is_query else item
        for item in texts
    ]

    vectors = _get_model().encode(
        model_inputs,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    if len(vectors) != len(texts):
        raise EmbeddingGenerationError(
            "The model returned an unexpected number of embeddings."
        )

    result: list[list[float]] = []

    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSION:
            raise EmbeddingGenerationError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings."
            )

        result.append([float(value) for value in vector])

    return result


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Generate embeddings for document chunks."""
    return _encode_texts(texts, is_query=False)


def embed_query(query: str) -> list[float]:
    """Generate an embedding for a search query."""
    if not isinstance(query, str) or not query.strip():
        raise EmbeddingInputError(
            "The search query must be a non-empty string."
        )

    return _encode_texts([query], is_query=True)[0]
