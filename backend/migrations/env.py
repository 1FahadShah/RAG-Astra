
from pathlib import Path
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

BACKEND_PATH = str(Path(__file__).resolve().parents[1])
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

from app.core.config import get_settings
from app.db.base import Base

from app.models import Document

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
target_metadata = Base.metadata

if settings.database_url:
    config.set_main_option(
        "sqlalchemy.url",
        settings.database_url.replace("%", "%%"),
    )


def run_migrations_offline() -> None:
    """Run migrations without creating a database connection."""
    url = config.get_main_option("sqlalchemy.url")

    if not url:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Set it in your local .env file before running migrations."
        )

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a live database connection."""
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            "Set it in your local .env file before running migrations."
        )

    connectable = engine_from_config(
        {"sqlalchemy.url": settings.database_url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
