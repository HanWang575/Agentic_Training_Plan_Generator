"""MCP tools for future function blocks.

The tools call the same service layer as FastAPI routes. They do not contain
separate retrieval or answer logic.
"""

from dataclasses import asdict

from sqlmodel import Session

from app.core.config import get_settings
from app.knowledge.service import KnowledgeService
from app.memory.service import MemoryService
from app.qa.service import QAService
from app.storage.database import engine, init_db


def create_mcp_server():
    """Create a FastMCP server with Q&A and knowledge-search tools."""

    from mcp.server.fastmcp import FastMCP

    init_db()
    mcp = FastMCP("personal-coach-evidence-qa")

    @mcp.tool()
    def ask_qa_assistant(question: str, topic_hint: str | None = None) -> dict:
        """Ask the evidence-based Q&A assistant.

        Input:
            question: user or block question.
            topic_hint: optional domain hint.
        Output:
            dict matching AskQuestionResponse.
        """

        with Session(engine) as session:
            service = QAService(session=session, settings=get_settings())
            return service.ask(question=question, topic_hint=topic_hint, debug=True).model_dump()

    @mcp.tool()
    def search_knowledge(query: str, topic_hint: str | None = None, top_k: int = 10) -> dict:
        """Search approved knowledge without generating an answer."""

        with Session(engine) as session:
            service = KnowledgeService(session=session, settings=get_settings())
            keyword, semantic, semantic_error = service.search(
                query=query,
                topic_hint=topic_hint,
                keyword_top_k=top_k,
                semantic_top_k=top_k,
            )
            return {
                "keyword_results": [asdict(item) for item in keyword],
                "semantic_results": [asdict(item) for item in semantic],
                "semantic_error": semantic_error,
            }

    @mcp.tool()
    def list_approved_sources() -> list[dict]:
        """List approved sources available to evidence-based answers."""

        with Session(engine) as session:
            service = KnowledgeService(session=session, settings=get_settings())
            return [item.model_dump() for item in service.list_sources(status="approved")]

    @mcp.tool()
    def get_answer_trace(trace_id: str) -> dict:
        """Return one stored Q&A trace by trace ID."""

        with Session(engine) as session:
            service = QAService(session=session, settings=get_settings())
            return service.get_trace(trace_id).model_dump()

    @mcp.tool()
    def get_user_memory_context(
        user_id: str,
        query: str = "",
        conversation_id: str | None = None,
        task_type: str = "qa",
    ) -> dict:
        """Return compact memory context for another function block."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            return service.get_memory_context(
                user_id=user_id,
                query=query,
                conversation_id=conversation_id,
                task_type=task_type,
                include_debug=True,
            ).model_dump()

    @mcp.tool()
    def search_user_episodes(user_id: str, query: str = "", top_k: int = 5) -> dict:
        """Search user episodic memories."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            return service.search_episodes(user_id=user_id, query=query, top_k=top_k).model_dump()

    @mcp.tool()
    def list_user_memory_items(
        user_id: str | None = None,
        status: str | None = None,
        category: str | None = None,
    ) -> list[dict]:
        """List user memory items for review or debugging."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            return [
                item.model_dump()
                for item in service.list_memory_items(
                    user_id=user_id,
                    status=status,
                    category=category,
                )
            ]

    @mcp.tool()
    def list_memory_key_proposals(
        user_id: str | None = None,
        review_status: str | None = None,
    ) -> list[dict]:
        """List proposed memory keys."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            return [
                item.model_dump()
                for item in service.list_key_proposals(
                    user_id=user_id,
                    review_status=review_status,
                )
            ]

    @mcp.tool()
    def get_recent_status(user_id: str, include_expired: bool = False) -> list[dict]:
        """Return recent status rows for a user."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            return [
                item.model_dump()
                for item in service.list_recent_status(
                    user_id=user_id,
                    include_expired=include_expired,
                )
            ]

    @mcp.tool()
    def list_expiring_recent_status(user_id: str, within_hours: int = 2) -> dict:
        """Return recent statuses that are close to expiration."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            return service.list_expiring_recent_status(
                user_id=user_id,
                within_hours=within_hours,
            ).model_dump()

    @mcp.tool()
    def get_active_plan_progress(user_id: str) -> dict | None:
        """Return the user's active plan progress if one exists."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            plan = service.get_active_plan(user_id=user_id)
            return plan.model_dump() if plan else None

    @mcp.tool()
    def get_training_log_summary(user_id: str, days: int = 30) -> dict:
        """Return a recent imported training-log summary."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            return service.get_training_log_summary(user_id=user_id, days=days).model_dump()

    return mcp


if __name__ == "__main__":
    create_mcp_server().run()
