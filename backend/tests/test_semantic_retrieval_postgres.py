
import os
import unittest
import uuid
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.models import Document, DocumentChunk, DocumentChunkEmbedding
from app.services.embedding_service import EMBEDDING_DIMENSION, MODEL_NAME
from app.services.semantic_retrieval import (
    RetrievalInputError,
    retrieve_chunks,
)


TEST_DATABASE_URL = os.getenv("RAG_ASTRA_TEST_DATABASE_URL")


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "Set RAG_ASTRA_TEST_DATABASE_URL to run PostgreSQL integration tests.",
)
class TestSemanticRetrievalPostgres(unittest.TestCase):
    """Semantic retrieval tests against isolated PostgreSQL schemas."""

    @classmethod
    def setUpClass(cls):
        url = make_url(os.environ["RAG_ASTRA_TEST_DATABASE_URL"])

        if url.drivername in {"postgresql", "postgres"}:
            url = url.set(drivername="postgresql+psycopg")

        cls.schema_name = f"rag_astra_test_{uuid.uuid4().hex}"
        cls.admin_engine = create_engine(url, pool_pre_ping=True)
        cls.engine = None

        try:
            with cls.admin_engine.begin() as connection:
                connection.execute(
                    text(f'CREATE SCHEMA "{cls.schema_name}"')
                )

            cls.engine = create_engine(
                url,
                connect_args={
                    "options": f"-csearch_path={cls.schema_name}"
                },
                pool_pre_ping=True,
            )

            alembic_config = Config("alembic.ini")
            migration_connection = cls.engine.connect()
            alembic_config.attributes["connection"] = migration_connection

            try:
                command.upgrade(alembic_config, "head")
            finally:
                migration_connection.close()

            cls.SessionFactory = sessionmaker(
                bind=cls.engine,
                autoflush=False,
                expire_on_commit=False,
            )

        except Exception:
            if cls.engine is not None:
                cls.engine.dispose()

            with cls.admin_engine.begin() as connection:
                connection.execute(
                    text(
                        f'DROP SCHEMA IF EXISTS '
                        f'"{cls.schema_name}" CASCADE'
                    )
                )

            cls.admin_engine.dispose()
            raise

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.dispose()

        if getattr(cls, "admin_engine", None) is not None:
            try:
                with cls.admin_engine.begin() as connection:
                    connection.execute(
                        text(
                            f'DROP SCHEMA IF EXISTS '
                            f'"{cls.schema_name}" CASCADE'
                        )
                    )
            finally:
                cls.admin_engine.dispose()

    def setUp(self):
        """Give each test a clean set of database records."""
        with self.SessionFactory() as session:
            session.execute(delete(DocumentChunkEmbedding))
            session.execute(delete(DocumentChunk))
            session.execute(delete(Document))
            session.commit()

    def add_result(
        self,
        session,
        *,
        filename,
        content,
        vector,
        status="ready",
        model_name=MODEL_NAME,
        page_number=1,
        section_title="Introduction",
    ):
        document = Document(
            id=uuid.uuid4(),
            filename=filename,
            content_type="text/plain",
            file_size_bytes=len(content.encode("utf-8")),
            checksum_sha256=uuid.uuid4().hex * 2,
            storage_key=f"{uuid.uuid4().hex}.txt",
            status=status,
        )
        session.add(document)
        session.flush()

        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=document.id,
            chunk_index=0,
            text=content,
            char_count=len(content),
            page_number=page_number,
            section_title=section_title,
        )
        session.add(chunk)
        session.flush()

        session.add(
            DocumentChunkEmbedding(
                chunk_id=chunk.id,
                model_name=model_name,
                embedding=vector,
            )
        )
        session.flush()

        return document.id, chunk.id

    @staticmethod
    def vector_with_first_two(first, second):
        vector = [0.0] * EMBEDDING_DIMENSION
        vector[0] = first
        vector[1] = second
        return vector

    def test_results_are_ranked_by_cosine_similarity(self):
        with self.SessionFactory() as session:
            _, exact_id = self.add_result(
                session,
                filename="exact.txt",
                content="Exact match",
                vector=self.vector_with_first_two(1.0, 0.0),
            )
            _, similar_id = self.add_result(
                session,
                filename="similar.txt",
                content="Similar match",
                vector=self.vector_with_first_two(0.8, 0.6),
            )
            _, orthogonal_id = self.add_result(
                session,
                filename="orthogonal.txt",
                content="Orthogonal match",
                vector=self.vector_with_first_two(0.0, 1.0),
            )
            session.commit()

            with patch(
                "app.services.semantic_retrieval.embed_query",
                return_value=self.vector_with_first_two(1.0, 0.0),
            ):
                results = retrieve_chunks(session, "example query")

            self.assertEqual(
                [result.chunk_id for result in results],
                [exact_id, similar_id, orthogonal_id],
            )
            self.assertAlmostEqual(
                results[0].similarity_score, 1.0, places=5
            )
            self.assertAlmostEqual(
                results[1].similarity_score, 0.8, places=5
            )
            self.assertAlmostEqual(
                results[2].similarity_score, 0.0, places=5
            )

    def test_limit_is_respected(self):
        with self.SessionFactory() as session:
            for index in range(3):
                self.add_result(
                    session,
                    filename=f"document-{index}.txt",
                    content=f"Content {index}",
                    vector=self.vector_with_first_two(1.0, 0.0),
                )
            session.commit()

            with patch(
                "app.services.semantic_retrieval.embed_query",
                return_value=self.vector_with_first_two(1.0, 0.0),
            ):
                results = retrieve_chunks(
                    session, "example query", limit=2
                )

            self.assertEqual(len(results), 2)

    def test_only_configured_model_and_ready_documents_are_searched(self):
        with self.SessionFactory() as session:
            _, current_model_id = self.add_result(
                session,
                filename="current.txt",
                content="Current model result",
                vector=self.vector_with_first_two(0.8, 0.6),
            )
            self.add_result(
                session,
                filename="other-model.txt",
                content="Other model result",
                vector=self.vector_with_first_two(1.0, 0.0),
                model_name="some-other-embedding-model",
            )
            self.add_result(
                session,
                filename="pending.txt",
                content="Pending document result",
                vector=self.vector_with_first_two(1.0, 0.0),
                status="pending",
            )
            session.commit()

            with patch(
                "app.services.semantic_retrieval.embed_query",
                return_value=self.vector_with_first_two(1.0, 0.0),
            ):
                results = retrieve_chunks(session, "example query")

            self.assertEqual(
                [result.chunk_id for result in results],
                [current_model_id],
            )

    def test_results_include_source_provenance(self):
        with self.SessionFactory() as session:
            document_id, chunk_id = self.add_result(
                session,
                filename="handbook.txt",
                content="Relevant handbook passage",
                vector=self.vector_with_first_two(1.0, 0.0),
                page_number=7,
                section_title="Security",
            )
            session.commit()

            with patch(
                "app.services.semantic_retrieval.embed_query",
                return_value=self.vector_with_first_two(1.0, 0.0),
            ):
                results = retrieve_chunks(session, "security")

            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.chunk_id, chunk_id)
            self.assertEqual(result.document_id, document_id)
            self.assertEqual(result.filename, "handbook.txt")
            self.assertEqual(result.text, "Relevant handbook passage")
            self.assertEqual(result.page_number, 7)
            self.assertEqual(result.section_title, "Security")
            self.assertIsInstance(result.similarity_score, float)

    def test_no_matches_returns_empty_list(self):
        with self.SessionFactory() as session:
            with patch(
                "app.services.semantic_retrieval.embed_query",
                return_value=self.vector_with_first_two(1.0, 0.0),
            ):
                results = retrieve_chunks(
                    session, "no matching documents"
                )

            self.assertEqual(results, [])

    def test_invalid_query_and_limit_are_rejected(self):
        with self.SessionFactory() as session:
            for query, limit in [
                ("", 5),
                ("   ", 5),
                ("valid query", 0),
                ("valid query", 101),
                ("valid query", True),
            ]:
                with self.subTest(query=query, limit=limit):
                    with self.assertRaises(RetrievalInputError):
                        retrieve_chunks(session, query, limit=limit)


if __name__ == "__main__":
    unittest.main()
