"""MCP tools for future function blocks.

The tools call the same service layer as FastAPI routes. They do not contain
separate retrieval or answer logic.
"""

from dataclasses import asdict

from sqlmodel import Session

from app.core.config import get_settings
from app.knowledge.service import KnowledgeService
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

    return mcp


if __name__ == "__main__":
    create_mcp_server().run()
