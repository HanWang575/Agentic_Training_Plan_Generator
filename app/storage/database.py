"""SQLite setup helpers.

The normal relational tables are defined with SQLModel. The keyword retrieval
index is a small SQLite FTS5 virtual table maintained by the indexing service.
"""

from collections.abc import Generator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import Settings, get_settings


def make_sqlite_url(sqlite_path: Path) -> str:
    """Input: a filesystem path. Output: a SQLAlchemy SQLite URL string."""

    return f"sqlite:///{sqlite_path.as_posix()}"


def create_app_engine(settings: Settings | None = None):
    """Create the SQLAlchemy engine used by SQLModel sessions.

    Parameters:
        settings: optional Settings object; defaults to the cached app settings.
    Returns:
        Engine: SQLAlchemy engine connected to `settings.sqlite_path`.
    """

    active_settings = settings or get_settings()
    active_settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        make_sqlite_url(active_settings.sqlite_path),
        connect_args={"check_same_thread": False},
    )


engine = create_app_engine()


def init_db(active_engine=None) -> None:
    """Create relational tables and the FTS5 table if they do not exist."""

    db_engine = active_engine or engine
    SQLModel.metadata.create_all(db_engine)
    with db_engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                source_id UNINDEXED,
                title,
                text,
                topic_tags
            )
            """
        )


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency.

    Yields:
        Session: one SQLModel session tied to the app engine.
    """

    with Session(engine) as session:
        yield session
