
import os
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.models import Document, DocumentChunk
from app.services.document_chunker import DocumentChunk as NewChunk
from app.services.document_parser import (
    ExtractionResult,
    ExtractedUnit,
)
from app.services.document_processing import (
    InvalidDocumentStateError,
    process_document,
)


TEST_DATABASE_URL = os.getenv("RAG_ASTRA_TEST_DATABASE_URL")


@unittest.skipUnless(
    TEST_DATABASE_URL,
    "Set RAG_ASTRA_TEST_DATABASE_URL to run PostgreSQL integration tests.",
)
class TestDocumentProcessingPostgres(unittest.TestCase):
    """Integration tests against an isolated PostgreSQL schema."""

    @classmethod
    def setUpClass(cls):
        url = make_url(os.environ["RAG_ASTRA_TEST_DATABASE_URL"])

        if url.drivername in {"postgresql", "postgres"}:
            url = url.set(drivername="postgresql+psycopg")

        cls.schema_name = f"rag_astra_test_{uuid.uuid4().hex}"

        cls.admin_engine = create_engine(
            url,
            pool_pre_ping=True,
        )

        cls.engine = None

        try:
            with cls.admin_engine.begin() as connection:
                connection.execute(
                    text(f'CREATE SCHEMA "{cls.schema_name}"')
                )

            # Scope all application tables and Alembic's version table
            # to the isolated schema.
            cls.engine = create_engine(
                url,
                connect_args={
                    "options": f"-csearch_path={cls.schema_name}"
                },
                pool_pre_ping=True,
            )

            alembic_config = Config("alembic.ini")
            migration_connection = cls.engine.connect()
            alembic_config.attributes["connection"] = (
                migration_connection
            )

            try:
                command.upgrade(alembic_config, "head")
            finally:
                migration_connection.close()

            cls.SessionFactory = sessionmaker(
                bind=cls.engine,
                autoflush=False,
                expire_on_commit=False,
            )

            cls.extraction = ExtractionResult(
                source_filename="integration-guide.txt",
                source_format="txt",
                units=(
                    ExtractedUnit(
                        text="PostgreSQL integration test content",
                        unit_index=0,
                    ),
                ),
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

    def create_document(self, status="pending"):
        document = Document(
            id=uuid.uuid4(),
            filename="integration-guide.txt",
            content_type="text/plain",
            file_size_bytes=100,
            checksum_sha256=uuid.uuid4().hex * 2,
            storage_key="integration-guide.txt",
            status=status,
        )

        with self.SessionFactory() as session:
            session.add(document)
            session.commit()

        return document.id

    def test_concurrent_requests_cannot_claim_same_document(self):
        document_id = self.create_document()

        extraction_started = Event()
        allow_extraction_to_finish = Event()

        def slow_extract(*args, **kwargs):
            extraction_started.set()

            if not allow_extraction_to_finish.wait(timeout=15):
                raise TimeoutError(
                    "Timed out waiting for the concurrency test."
                )

            return self.extraction

        def process_in_own_session():
            with self.SessionFactory() as session:
                return process_document(session, document_id)

        with (
            patch(
                "app.services.document_processing.extract_document",
                side_effect=slow_extract,
            ),
            patch(
                "app.services.document_processing.chunk_document",
                return_value=[
                    NewChunk(
                        chunk_index=0,
                        text="PostgreSQL integration test content",
                    )
                ],
            ),
            patch(
                "app.services.document_processing.embed_documents",
                return_value=[[0.1] * 384],
            ),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            first_request = executor.submit(process_in_own_session)

            try:
                self.assertTrue(
                    extraction_started.wait(timeout=10),
                    "The first request did not reach extraction.",
                )

                # The first request has committed "processing" and is
                # paused during extraction.
                with self.SessionFactory() as second_session:
                    with self.assertRaises(InvalidDocumentStateError):
                        process_document(second_session, document_id)

            finally:
                allow_extraction_to_finish.set()

            first_request.result(timeout=20)

        with self.SessionFactory() as session:
            document = session.get(Document, document_id)
            chunks = session.scalars(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == document_id
                )
            ).all()

            self.assertEqual(document.status, "ready")
            self.assertEqual(len(chunks), 1)
            self.assertEqual(
                chunks[0].text,
                "PostgreSQL integration test content",
            )

    def test_failed_replacement_preserves_existing_chunks(self):
        document_id = self.create_document(status="failed")

        with self.SessionFactory() as session:
            session.add(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=0,
                    text="Previously stored content",
                    char_count=len("Previously stored content"),
                )
            )
            session.commit()

        # Duplicate indices trigger PostgreSQL's unique constraint
        # after old chunks are deleted in the uncommitted transaction.
        duplicate_chunks = [
            NewChunk(chunk_index=0, text="Replacement A"),
            NewChunk(chunk_index=0, text="Replacement B"),
        ]

        with (
            patch(
                "app.services.document_processing.extract_document",
                return_value=self.extraction,
            ),
            patch(
                "app.services.document_processing.chunk_document",
                return_value=duplicate_chunks,
            ),
            self.assertRaises(IntegrityError),
        ):
            with self.SessionFactory() as session:
                process_document(session, document_id)

        # Verify that rollback preserved the original chunks.
        with self.SessionFactory() as session:
            document = session.get(Document, document_id)
            chunks = session.scalars(
                select(DocumentChunk).where(
                    DocumentChunk.document_id == document_id
                )
            ).all()

            self.assertEqual(document.status, "failed")
            self.assertEqual(len(chunks), 1)
            self.assertEqual(
                chunks[0].text,
                "Previously stored content",
            )


if __name__ == "__main__":
    unittest.main()
