"""Database session management and engine initialization for Haddu.

Supports PostgreSQL + PostGIS with connection pooling and automated extension setup.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Load environment variables from .env if available
load_dotenv()

DEFAULT_DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'haddu')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'haddu_secret')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'haddu_db')}"
)

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine(url: str | None = None) -> Engine:
    """Create and return a SQLAlchemy engine with connection health checks."""
    target_url = url or DATABASE_URL
    return create_engine(
        target_url,
        pool_pre_ping=True,
        echo=os.getenv("SQL_ECHO", "false").lower() in ("true", "1", "yes"),
    )


# Default engine and sessionmaker
engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db(session_factory=SessionLocal) -> Generator[Session, None, None]:
    """Context manager for transactional database sessions with auto-commit/rollback."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI database dependency yielding an active transactional session."""
    with get_db() as session:
        yield session


def init_db(target_engine: Engine | None = None) -> None:
    """Initialize database extensions and create all declared tables."""
    from haddu.db.models import Base

    eng = target_engine or engine

    # Enable PostGIS extension for PostgreSQL databases
    if eng.dialect.name == "postgresql":
        with eng.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
            conn.commit()

    Base.metadata.create_all(bind=eng)

    if eng.dialect.name == "postgresql":
        with eng.connect() as conn:
            conn.execute(text("""
                ALTER TABLE encroachment_incidents ADD COLUMN IF NOT EXISTS violation_tier VARCHAR(50);
                ALTER TABLE encroachment_incidents ADD COLUMN IF NOT EXISTS encroached_rf_name VARCHAR(255);
                ALTER TABLE encroachment_incidents ADD COLUMN IF NOT EXISTS rf_area_acres DOUBLE PRECISION;
                ALTER TABLE encroachment_incidents ADD COLUMN IF NOT EXISTS riparian_breach BOOLEAN DEFAULT FALSE;
                ALTER TABLE encroachment_incidents ADD COLUMN IF NOT EXISTS sensor VARCHAR(20);
            """))
            conn.commit()
