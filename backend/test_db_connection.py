
from sqlalchemy import create_engine, text

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is missing from .env")

    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10},
    )

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database(), current_user")
            )
            database, user = result.one()

            print("Connection successful!")
            print(f"Database: {database}")
            print(f"User: {user}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
