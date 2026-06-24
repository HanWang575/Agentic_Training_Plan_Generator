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
        _ensure_column(
            connection,
            table_name="source_documents",
            column_name="parser_warnings_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            connection,
            table_name="source_documents",
            column_name="parser_progress",
            column_definition="REAL NOT NULL DEFAULT 0.0",
        )
        _ensure_column(
            connection,
            table_name="source_documents",
            column_name="parser_progress_message",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="source_documents",
            column_name="parser_options_json",
            column_definition="TEXT NOT NULL DEFAULT '{}'",
        )
        _ensure_column(
            connection,
            table_name="knowledge_chunks",
            column_name="quality_warnings_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            connection,
            table_name="knowledge_chunks",
            column_name="content_type",
            column_definition="TEXT NOT NULL DEFAULT 'plain_text'",
        )
        _ensure_column(
            connection,
            table_name="knowledge_chunks",
            column_name="content_metadata_json",
            column_definition="TEXT NOT NULL DEFAULT '{}'",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="merged_chunk_ids_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="planner_used",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="user_id",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="conversation_id",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="clean_question",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="clarified_question",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="search_query",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="query_terms_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="planner_notes_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="memory_context_json",
            column_definition="TEXT NOT NULL DEFAULT '{}'",
        )
        _ensure_column(
            connection,
            table_name="answer_traces",
            column_name="memory_debug_ids_json",
            column_definition="TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            connection,
            table_name="conversations",
            column_name="rolling_summary_until_message_id",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="training_sessions",
            column_name="title",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="training_sessions",
            column_name="start_at",
            column_definition="DATETIME",
        )
        _ensure_column(
            connection,
            table_name="training_sessions",
            column_name="end_at",
            column_definition="DATETIME",
        )
        _ensure_column(
            connection,
            table_name="training_sessions",
            column_name="duration_seconds",
            column_definition="INTEGER",
        )
        _ensure_column(
            connection,
            table_name="episodes",
            column_name="program_context_id",
            column_definition="TEXT",
        )
        _ensure_column(
            connection,
            table_name="training_plan_progress",
            column_name="version_id",
            column_definition="TEXT",
        )
        _ensure_column(connection, "exercise_prescriptions", "volume", "REAL")
        _ensure_column(connection, "exercise_prescriptions", "volume_unit", "TEXT")
        _ensure_column(
            connection,
            "exercise_prescriptions",
            "calculation_trace_ids_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            connection,
            "program_versions",
            "source_references_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        _ensure_column(
            connection,
            "training_plan_progress",
            "plan_snapshot_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        _ensure_column(connection, "program_requests", "goal", "TEXT")
        _ensure_column(
            connection,
            "program_requests",
            "constraints_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        _ensure_column(
            connection,
            "program_sessions",
            "duration_calculation_trace_id",
            "TEXT",
        )
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


def _ensure_column(connection, table_name: str, column_name: str, column_definition: str) -> None:
    """Add a SQLite column when opening an older local database."""

    columns = connection.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    if any(column[1] == column_name for column in columns):
        return
    connection.exec_driver_sql(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    )


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency.

    Yields:
        Session: one SQLModel session tied to the app engine.
    """

    with Session(engine) as session:
        yield session
