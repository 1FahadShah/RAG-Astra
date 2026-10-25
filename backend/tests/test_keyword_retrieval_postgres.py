import os
import unittest
import uuid
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from app.models import Document, DocumentChunk
from app.services.semantic_retrieval import RetrievalInputError
from app.services.keyword_retrieval import retrieve_keyword_chunks


TEST_DATABASE_URL = os.getenv("RAG_ASTRA_TEST_DATABASE_URL")


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "Set RAG_ASTRA_TEST_DATABASE_URL to run PostgreSQL integration tests.",
)
class TestKeywordRetrievalPostgres(unittest.TestCase):
    """Keyword retrieval tests using isolated PostgreSQL schemas."""

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

            config = Config("alembic.ini")
            migration_connection = cls.engine.connect()
            config.attributes["connection"] = migration_connection

            try:
                command.upgrade(config, "head")
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
        with self.SessionFactory() as session:
            session.execute(delete(DocumentChunk))
            session.execute(delete(Document))
            session.commit()

    def add_chunk(
        self,
        session,
        *,
        filename,
        content,
        status="ready",
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

        return document.id, chunk.id

    def test_keyword_search_returns_matching_chunk_and_provenance(self):
        with self.SessionFactory() as session:
            document_id, chunk_id = self.add_chunk(
                session,
                filename="security-handbook.txt",
                content="Security controls protect enterprise data.",
                page_number=7,
                section_title="Security",
            )
            session.commit()

            results = retrieve_keyword_chunks(session, "security controls")

            self.assertEqual(len(results), 1)

            result = results[0]
            self.assertEqual(result.chunk_id, chunk_id)
            self.assertEqual(result.document_id, document_id)
            self.assertEqual(result.filename, "security-handbook.txt")
            self.assertEqual(
                result.text,
                "Security controls protect enterprise data.",
            )
            self.assertEqual(result.page_number, 7)
            self.assertEqual(result.section_title, "Security")
            self.assertIsInstance(result.keyword_score, float)
            self.assertGreater(result.keyword_score, 0.0)

    def test_only_ready_documents_are_searched(self):
        with self.SessionFactory() as session:
            _, ready_id = self.add_chunk(
                session,
                filename="ready.txt",
                content="Enterprise security policies.",
                status="ready",
            )
            self.add_chunk(
                session,
                filename="pending.txt",
                content="Enterprise security policies.",
                status="pending",
            )
            session.commit()

            results = retrieve_keyword_chunks(
                session,
                "enterprise security",
            )

            self.assertEqual(
                [result.chunk_id for result in results],
                [ready_id],
            )

    def test_limit_is_respected(self):
        with self.SessionFactory() as session:
            for index in range(3):
                self.add_chunk(
                    session,
                    filename=f"security-{index}.txt",
                    content=f"Security policy document number {index}.",
                )
            session.commit()

            results = retrieve_keyword_chunks(
                session,
                "security policy",
                limit=2,
            )

            self.assertEqual(len(results), 2)

    def test_results_are_ranked_by_keyword_score(self):
        with self.SessionFactory() as session:
            self.add_chunk(
                session,
                filename="repeated.txt",
                content="Security security security security policy.",
            )
            self.add_chunk(
                session,
                filename="single.txt",
                content="Security policy.",
            )
            session.commit()

            results = retrieve_keyword_chunks(session, "security")

            self.assertEqual(len(results), 2)
            self.assertGreater(
                results[0].keyword_score,
                results[1].keyword_score,
            )

    def test_no_matches_returns_empty_list(self):
        with self.SessionFactory() as session:
            results = retrieve_keyword_chunks(
                session,
                "unrelated terminology",
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
                        retrieve_keyword_chunks(
                            session,
                            query,
                            limit=limit,
                        )


if __name__ == "__main__":
    unittest.main()