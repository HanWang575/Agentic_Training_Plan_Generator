"""FastAPI app factory and local server entry point."""

from fastapi import FastAPI

from app.api.routes_knowledge import router as knowledge_router
from app.api.routes_memory import router as memory_router
from app.api.routes_qa import router as qa_router
from app.storage.database import init_db


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: app with Q&A and knowledge routes mounted.
    """

    init_db()
    app = FastAPI(title="Personal Coach Evidence-Based Q&A Assistant")
    app.include_router(qa_router)
    app.include_router(knowledge_router)
    app.include_router(memory_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Simple readiness check for local development."""

        return {"status": "ok"}

    return app


app = create_app()
