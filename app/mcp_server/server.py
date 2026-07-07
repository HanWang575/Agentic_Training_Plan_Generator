"""MCP tools for future function blocks.

The tools call the same service layer as FastAPI routes. They do not contain
separate retrieval or answer logic.
"""

from dataclasses import asdict

from sqlmodel import Session

from app.core.config import get_settings
from app.knowledge.service import KnowledgeService
from app.memory.service import MemoryService
from app.memory.schemas import PlanActivationRequest
from app.program.calculators import CalculatorService
from app.program.schemas import ProgramReviewRequest, ProgramTurnRequest, XunjiWriteRequest
from app.program.service import ProgramService
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
        """Return memory context for another function block."""

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
    def activate_approved_plan(
        user_id: str,
        program_id: str,
        version_id: str,
        version_number: int,
        planned_sessions: list[dict],
        plan_snapshot: dict,
    ) -> dict:
        """Atomically activate an approved plan and deactivate older plans."""

        with Session(engine) as session:
            service = MemoryService(session=session, settings=get_settings())
            return service.activate_plan(
                PlanActivationRequest(
                    user_id=user_id,
                    program_id=program_id,
                    version_id=version_id,
                    version_number=version_number,
                    planned_sessions=planned_sessions,
                    plan_snapshot=plan_snapshot,
                )
            ).model_dump(mode="json")

    @mcp.tool()
    def send_program_conversation_turn(
        user_id: str,
        conversation_id: str,
        message: str,
        expected_context_revision: int | None = None,
    ) -> dict:
        """Process one Program Generator conversation turn."""

        with Session(engine) as session:
            settings = get_settings()
            service = ProgramService(session=session, settings=settings)
            return service.process_turn(
                conversation_id,
                ProgramTurnRequest(
                    user_id=user_id,
                    message=message,
                    expected_context_revision=expected_context_revision,
                ),
            ).model_dump(mode="json")

    @mcp.tool()
    def get_program_context(context_id: str) -> dict:
        """Return one checkpointed ProgramContext."""

        with Session(engine) as session:
            return ProgramService(session=session, settings=get_settings()).get_context(context_id).model_dump(mode="json")

    @mcp.tool()
    def get_program_version(program_id: str, version_id: str) -> dict:
        """Return a complete generated program version."""

        with Session(engine) as session:
            return ProgramService(session=session, settings=get_settings()).get_version(
                program_id, version_id
            ).model_dump(mode="json")

    @mcp.tool()
    def review_training_program(
        user_id: str,
        program_id: str,
        version_id: str,
        decision: str,
        reason: str | None = None,
        expected_version_number: int | None = None,
    ) -> dict:
        """Approve and activate, or reject, one generated version."""

        with Session(engine) as session:
            return ProgramService(session=session, settings=get_settings()).review_program(
                program_id,
                version_id,
                ProgramReviewRequest(
                    user_id=user_id,
                    decision=decision,
                    reason=reason,
                    expected_version_number=expected_version_number,
                ),
            ).model_dump(mode="json")

    @mcp.tool()
    def get_program_trace(trace_id: str) -> dict:
        """Return one complete Program Generator trace."""

        with Session(engine) as session:
            return ProgramService(session=session, settings=get_settings()).get_trace(trace_id).model_dump(mode="json")

    @mcp.tool()
    def write_training_program_to_xunji(
        user_id: str,
        program_id: str,
        version_id: str,
        confirmed: bool,
    ) -> dict:
        """Attempt separately approved Xunji write-back."""

        with Session(engine) as session:
            return ProgramService(session=session, settings=get_settings()).write_to_xunji(
                program_id,
                version_id,
                XunjiWriteRequest(user_id=user_id, confirmed=confirmed),
            ).model_dump(mode="json")

    def calculator(tool_name: str, inputs: dict) -> dict:
        return CalculatorService(get_settings()).run(tool_name, inputs).model_dump(mode="json")

    @mcp.tool()
    def calculate_expression(expression: str, unit: str = "count") -> dict:
        return calculator("calculate_expression", {"expression": expression, "unit": unit})

    @mcp.tool()
    def estimate_repetition_max(weight: float, reps: int, method: str = "epley", unit: str = "kg") -> dict:
        return calculator("estimate_repetition_max", {"weight": weight, "reps": reps, "method": method, "unit": unit})

    @mcp.tool()
    def calculate_percentage_load(one_rm: float, percentage: float, unit: str = "kg") -> dict:
        return calculator("calculate_percentage_load", {"one_rm": one_rm, "percentage": percentage, "unit": unit})

    @mcp.tool()
    def round_training_load(load: float, increment: float = 0.5, unit: str = "kg") -> dict:
        return calculator("round_training_load", {"load": load, "increment": increment, "unit": unit})

    @mcp.tool()
    def calculate_volume_load(sets: int, reps: int, load: float, unit: str = "kg") -> dict:
        return calculator("calculate_volume_load", {"sets": sets, "reps": reps, "load": load, "unit": unit})

    @mcp.tool()
    def calculate_session_duration(work_seconds: float, rest_seconds: float, transition_seconds: float = 0) -> dict:
        return calculator(
            "calculate_session_duration",
            {"work_seconds": work_seconds, "rest_seconds": rest_seconds, "transition_seconds": transition_seconds},
        )

    @mcp.tool()
    def calculate_rest_interval(goal: str, compound: bool = True) -> dict:
        return calculator("calculate_rest_interval", {"goal": goal, "compound": compound})

    @mcp.tool()
    def calculate_weekly_frequency(sessions: int, weeks: int = 1) -> dict:
        return calculator("calculate_weekly_frequency", {"sessions": sessions, "weeks": weeks})

    @mcp.tool()
    def calculate_progression_increment(current_value: float, percentage: float, unit: str = "count") -> dict:
        return calculator(
            "calculate_progression_increment",
            {"current_value": current_value, "percentage": percentage, "unit": unit},
        )

    return mcp


if __name__ == "__main__":
    create_mcp_server().run()
