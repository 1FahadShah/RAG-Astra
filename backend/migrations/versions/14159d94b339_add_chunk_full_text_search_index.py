"""add chunk full text search index

Revision ID: 14159d94b339
Revises: b2a3b5dbe544
Create Date: 2026-10-25 14:46:39.580435
"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "14159d94b339"
down_revision: Union[str, Sequence[str], None] = "b2a3b5dbe544"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


INDEX_NAME = "ix_document_chunks_text_search"


def upgrade() -> None:
    """Create a GIN index for English full-text search."""
    op.execute(
        f"""
        CREATE INDEX {INDEX_NAME}
        ON document_chunks
        USING GIN (to_tsvector('english', text))
        """
    )


def downgrade() -> None:
    """Remove the full-text search index."""
    op.drop_index(
        INDEX_NAME,
        table_name="document_chunks",
    )