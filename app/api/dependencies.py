"""FastAPI dependencies for settings and services."""

from collections.abc import Generator

from app.core.config import Settings, get_settings
from app.memory.service import MemoryService
from app.qa.service import QAService
from app.storage.database import get_session


def get_app_settings() -> Settings:
    """Return cached app settings for dependency injection."""

    return get_settings()


def qa_service_dependency() -> Generator[QAService, None, None]:
    """Yield a QAService with a managed database session."""

    settings = get_settings()
    for session in get_session():
        memory_service = MemoryService(session=session, settings=settings)
        yield QAService(session=session, settings=settings, memory_service=memory_service)


def memory_service_dependency() -> Generator[MemoryService, None, None]:
    """Yield a MemoryService with a managed database session."""

    settings = get_settings()
    for session in get_session():
        yield MemoryService(session=session, settings=settings)
