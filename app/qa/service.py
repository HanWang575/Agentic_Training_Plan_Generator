"""High-level Q&A service used by FastAPI, Streamlit, and MCP."""

import json
import time
from uuid import uuid4

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.llm_provider import OpenAIProvider
from app.knowledge.service import KnowledgeService
from app.memory.service import MemoryService
from app.qa.graph import run_langgraph_or_nodes
from app.qa.nodes import (
    AnswerGenerator,
    CitationChecker,
    ContextBuilder,
    InputValidator,
    NeighborExpander,
    QueryPlanner,
    ResultMerger,
    RetrievalNode,
    SafetyClassifier,
)
from app.qa.schemas import AnswerTraceOut, AskQuestionResponse, CitationOut, QAState
from app.storage.models import AnswerTrace, Citation, SafetyFlag


class QAService:
    """Public service interface for evidence-grounded answers and trace lookup."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        knowledge_service: KnowledgeService | None = None,
        llm_provider: OpenAIProvider | None = None,
        memory_service: MemoryService | None = None,
    ):
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider or OpenAIProvider(settings)
        self.knowledge_service = knowledge_service or KnowledgeService(
            session=session,
            settings=settings,
            llm_provider=self.llm_provider,
        )
        self.memory_service = memory_service

    def ask(
        self,
        question: str,
        topic_hint: str | None = None,
        debug: bool = False,
        user_id: str | None = None,
        conversation_id: str | None = None,
        include_memory: bool = False,
        memory_debug: bool = False,
    ) -> AskQuestionResponse:
        """Run the Q&A workflow.

        Parameters:
            question: raw user question string.
            topic_hint: optional domain hint such as `nutrition` or `powerlifting`.
            debug: when true, include retrieved chunk IDs and timing details.
        Returns:
            AskQuestionResponse: answer, citations, safety note, trace ID, and optional debug data.
        """

        started = time.perf_counter()
        memory_payload, memory_errors, resolved_conversation_id = self._prepare_memory_context(
            question=question,
            user_id=user_id,
            conversation_id=conversation_id,
            include_memory=include_memory,
            memory_debug=memory_debug,
        )
        initial_state: QAState = {
            "question": question,
            "topic_hint": topic_hint,
            "user_id": user_id,
            "conversation_id": resolved_conversation_id,
            "memory_context_rendered": memory_payload.get("rendered_context", ""),
            "memory_context_payload": memory_payload,
            "memory_debug_ids": memory_payload.get("memory_ids", []),
            "errors": memory_errors,
            "timings": {},
        }
        nodes = [
            InputValidator(),
            SafetyClassifier(),
            QueryPlanner(self.llm_provider),
            RetrievalNode(self.knowledge_service),
            ResultMerger(self.settings),
            NeighborExpander(self.knowledge_service),
            ContextBuilder(self.settings),
            AnswerGenerator(self.settings, self.llm_provider),
            CitationChecker(),
        ]
        state = run_langgraph_or_nodes(initial_state, nodes)
        total_latency_ms = int((time.perf_counter() - started) * 1000)
        trace_id = self._store_trace(state, total_latency_ms)
        self._store_assistant_message(
            conversation_id=resolved_conversation_id,
            answer=state.get("final_answer", ""),
            trace_id=trace_id,
        )
        citations = state.get("citations", [])
        safety_flags = state.get("safety_flags", [])
        debug_payload = None
        if debug or memory_debug:
            debug_payload = {
                "planner_used": state.get("planner_used"),
                "clean_question": state.get("clean_question"),
                "clarified_question": state.get("clarified_question"),
                "search_query": state.get("search_query"),
                "query_terms": state.get("query_terms", []),
                "planner_notes": state.get("planner_notes", []),
                "keyword_chunk_ids": [chunk.chunk_id for chunk in state.get("keyword_results", [])],
                "semantic_chunk_ids": [chunk.chunk_id for chunk in state.get("semantic_results", [])],
                "merged_chunk_ids": [chunk.chunk_id for chunk in state.get("merged_results", [])],
                "final_chunk_ids": [chunk.chunk_id for chunk in state.get("context_results", [])],
                "explicit_citation_chunk_ids": [
                    citation.chunk_id for citation in state.get("context_citations", [])
                ],
                "returned_citation_chunk_ids": [citation.chunk_id for citation in citations],
                "semantic_error": state.get("semantic_error"),
                "memory_context": state.get("memory_context_payload", {}),
                "memory_debug_ids": state.get("memory_debug_ids", []),
                "errors": state.get("errors", []),
                "timings": state.get("timings", {}),
            }
        return AskQuestionResponse(
            answer=state.get("final_answer", ""),
            source_ids=sorted({citation.source_id for citation in citations}),
            citations=citations,
            safety_note=_safety_note(safety_flags),
            insufficient_evidence=state.get("insufficient_evidence", False),
            trace_id=trace_id,
            conversation_id=resolved_conversation_id,
            debug=debug_payload,
        )

    def _prepare_memory_context(
        self,
        question: str,
        user_id: str | None,
        conversation_id: str | None,
        include_memory: bool,
        memory_debug: bool,
    ) -> tuple[dict, list[str], str | None]:
        """Fetch optional memory context and log the user turn when possible."""

        errors: list[str] = []
        resolved_conversation_id = (conversation_id or "").strip() or None
        if self.memory_service and user_id:
            try:
                conversation = self.memory_service.create_conversation(
                    user_id=user_id,
                    conversation_id=resolved_conversation_id,
                    title="Q&A chat",
                    topic="qa",
                )
                resolved_conversation_id = conversation.conversation_id
                self.memory_service.append_message(
                    conversation_id=resolved_conversation_id,
                    role="user",
                    content=question,
                    source_block="qa",
                )
            except Exception as exc:
                errors.append(f"Memory message append failed: {exc}")

        if not include_memory:
            return {}, errors, resolved_conversation_id
        if not user_id:
            errors.append("include_memory=true was ignored because user_id was not provided.")
            return {}, errors, resolved_conversation_id
        if self.memory_service is None:
            errors.append("include_memory=true was ignored because MemoryService is unavailable.")
            return {}, errors, resolved_conversation_id
        try:
            context = self.memory_service.get_memory_context(
                user_id=user_id,
                query=question,
                conversation_id=resolved_conversation_id,
                task_type="qa",
                include_debug=memory_debug,
            )
            return context.model_dump(mode="json"), errors, resolved_conversation_id
        except Exception as exc:
            errors.append(f"Memory context retrieval failed: {exc}")
            return {}, errors, resolved_conversation_id

    def _store_assistant_message(
        self,
        conversation_id: str | None,
        answer: str,
        trace_id: str,
    ) -> None:
        """Best-effort logging of the assistant answer into working memory."""

        if not self.memory_service or not conversation_id:
            return
        try:
            self.memory_service.append_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                source_block="qa",
                qa_trace_id=trace_id,
            )
        except Exception:
            return

    def get_trace(self, trace_id: str) -> AnswerTraceOut:
        """Load one stored answer trace with citations and safety flags."""

        trace = self.session.get(AnswerTrace, trace_id)
        if trace is None:
            raise NotFoundError(f"Trace not found: {trace_id}")
        citations = self.session.exec(select(Citation).where(Citation.trace_id == trace_id)).all()
        safety_flags = self.session.exec(select(SafetyFlag).where(SafetyFlag.trace_id == trace_id)).all()
        return AnswerTraceOut(
            trace_id=trace.trace_id,
            question=trace.question,
            topic_hint=trace.topic_hint,
            user_id=trace.user_id,
            conversation_id=trace.conversation_id,
            safety_mode=trace.safety_mode,
            planner_used=trace.planner_used,
            clean_question=trace.clean_question,
            clarified_question=trace.clarified_question,
            search_query=trace.search_query,
            query_terms=json.loads(trace.query_terms_json or "[]"),
            planner_notes=json.loads(trace.planner_notes_json or "[]"),
            keyword_chunk_ids=json.loads(trace.keyword_chunk_ids_json or "[]"),
            semantic_chunk_ids=json.loads(trace.semantic_chunk_ids_json or "[]"),
            merged_chunk_ids=json.loads(trace.merged_chunk_ids_json or "[]"),
            final_chunk_ids=json.loads(trace.final_chunk_ids_json or "[]"),
            answer=trace.answer,
            total_latency_ms=trace.total_latency_ms,
            error=trace.error,
            timings=json.loads(trace.timings_json or "{}"),
            memory_context=json.loads(trace.memory_context_json or "{}"),
            memory_debug_ids=json.loads(trace.memory_debug_ids_json or "[]"),
            citations=[
                CitationOut(
                    source_id=item.source_id,
                    chunk_id=item.chunk_id,
                    title=item.citation_text,
                    page_or_section=item.answer_section,
                )
                for item in citations
            ],
            safety_flags=[
                {
                    "flag_type": item.flag_type,
                    "severity": item.severity,
                    "explanation": item.explanation,
                }
                for item in safety_flags
            ],
        )

    def _store_trace(self, state: QAState, total_latency_ms: int) -> str:
        """Persist trace, citation, and safety rows for one graph run."""

        trace_id = f"trc_{uuid4().hex[:12]}"
        citations: list[CitationOut] = state.get("citations", [])
        safety_flags = state.get("safety_flags", [])
        errors = state.get("errors", [])
        self.session.add(
            AnswerTrace(
                trace_id=trace_id,
                question=state.get("question", ""),
                topic_hint=state.get("topic_hint"),
                user_id=state.get("user_id"),
                conversation_id=state.get("conversation_id"),
                safety_mode=state.get("safety_mode", "normal"),
                planner_used=state.get("planner_used"),
                clean_question=state.get("clean_question"),
                clarified_question=state.get("clarified_question"),
                search_query=state.get("search_query"),
                query_terms_json=json.dumps(state.get("query_terms", [])),
                planner_notes_json=json.dumps(state.get("planner_notes", [])),
                keyword_chunk_ids_json=json.dumps(
                    [chunk.chunk_id for chunk in state.get("keyword_results", [])]
                ),
                semantic_chunk_ids_json=json.dumps(
                    [chunk.chunk_id for chunk in state.get("semantic_results", [])]
                ),
                merged_chunk_ids_json=json.dumps(
                    [chunk.chunk_id for chunk in state.get("merged_results", [])]
                ),
                final_chunk_ids_json=json.dumps(
                    [chunk.chunk_id for chunk in state.get("context_results", [])]
                ),
                answer=state.get("final_answer", ""),
                memory_context_json=json.dumps(state.get("memory_context_payload", {})),
                memory_debug_ids_json=json.dumps(state.get("memory_debug_ids", [])),
                total_latency_ms=total_latency_ms,
                error="; ".join(errors) if errors else None,
                timings_json=json.dumps(state.get("timings", {})),
            )
        )
        for citation in citations:
            self.session.add(
                Citation(
                    citation_id=f"cit_{uuid4().hex[:12]}",
                    trace_id=trace_id,
                    source_id=citation.source_id,
                    chunk_id=citation.chunk_id,
                    answer_section=citation.page_or_section or "",
                    citation_text=citation.title,
                )
            )
        for flag in safety_flags:
            self.session.add(
                SafetyFlag(
                    flag_id=f"safe_{uuid4().hex[:12]}",
                    trace_id=trace_id,
                    flag_type=flag["flag_type"],
                    severity=flag["severity"],
                    explanation=flag["explanation"],
                )
            )
        self.session.commit()
        return trace_id


def _safety_note(safety_flags: list[dict[str, str]]) -> str | None:
    """Return a readable safety note when safety flags were raised."""

    if not safety_flags:
        return None
    categories = ", ".join(flag["flag_type"] for flag in safety_flags)
    return (
        f"Safety caution triggered for: {categories}. This assistant provides general "
        "fitness education, not diagnosis, treatment, or unsafe performance advice."
    )
