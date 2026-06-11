"""High-level Q&A service used by FastAPI, Streamlit, and MCP."""

import json
import time
from uuid import uuid4

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import NotFoundError
from app.core.llm_provider import OpenAIProvider
from app.knowledge.service import KnowledgeService
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
    ):
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider or OpenAIProvider(settings)
        self.knowledge_service = knowledge_service or KnowledgeService(
            session=session,
            settings=settings,
            llm_provider=self.llm_provider,
        )

    def ask(self, question: str, topic_hint: str | None = None, debug: bool = False) -> AskQuestionResponse:
        """Run the Q&A workflow.

        Parameters:
            question: raw user question string.
            topic_hint: optional domain hint such as `nutrition` or `powerlifting`.
            debug: when true, include retrieved chunk IDs and timing details.
        Returns:
            AskQuestionResponse: answer, citations, safety note, trace ID, and optional debug data.
        """

        started = time.perf_counter()
        initial_state: QAState = {
            "question": question,
            "topic_hint": topic_hint,
            "errors": [],
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
        citations = state.get("citations", [])
        safety_flags = state.get("safety_flags", [])
        debug_payload = None
        if debug:
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
            debug=debug_payload,
        )

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
