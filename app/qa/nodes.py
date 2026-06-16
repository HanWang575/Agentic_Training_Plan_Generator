"""Small Q&A workflow nodes.

Each node receives a QAState dictionary and returns only the fields it changes.
This keeps the graph easy to debug and easy to replace with LangGraph runtime.
"""

import re
import time

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.llm_provider import AnswerRequest, OpenAIProvider, QueryPlanRequest
from app.knowledge.retrieval import expand_query_terms, normalize_query
from app.knowledge.schemas import RetrievedChunk
from app.knowledge.service import KnowledgeService
from app.qa.schemas import CitationOut, QAState


def _timed(node_name: str, state: QAState, work):
    """Run `work` and store elapsed milliseconds under `timings[node_name]`."""

    started = time.perf_counter()
    updates = work()
    timings = dict(state.get("timings", {}))
    timings[node_name] = int((time.perf_counter() - started) * 1000)
    updates["timings"] = timings
    return updates


class InputValidator:
    """Validate and normalize the user's question."""

    def __call__(self, state: QAState) -> QAState:
        """Input: `question`, optional `topic_hint`. Output: `clean_question` or errors."""

        def work():
            question = normalize_query(state.get("question", ""))
            errors = list(state.get("errors", []))
            if not question:
                errors.append("Question is empty.")
            return {"clean_question": question, "errors": errors}

        return _timed("InputValidator", state, work)


class SafetyClassifier:
    """Rule-based v1 safety classifier with clear flags."""

    FLAG_RULES = {
        "medical": ["diagnose", "disease", "chest pain", "blood pressure", "diabetes"],
        "injury": ["injury", "pain", "sprain", "strain", "rehab"],
        "doping": ["steroid", "testosterone", "wada", "banned substance", "sarms"],
        "eating_disorder": ["binge", "purge", "anorexia", "eating disorder"],
        "extreme_diet": ["starve", "water fast", "800 calories", "rapid weight loss"],
    }

    def __call__(self, state: QAState) -> QAState:
        """Input: `clean_question`. Output: `safety_flags`, `safety_mode`."""

        def work():
            question = state.get("clean_question", "").lower()
            flags: list[dict[str, str]] = []
            for flag_type, needles in self.FLAG_RULES.items():
                if any(needle in question for needle in needles):
                    flags.append(
                        {
                            "flag_type": flag_type,
                            "severity": "caution",
                            "explanation": f"Question matched {flag_type} safety terms.",
                        }
                    )
            safety_mode = "caution" if flags else "normal"
            return {"safety_flags": flags, "safety_mode": safety_mode}

        return _timed("SafetyClassifier", state, work)


class QueryPlanner:
    """Prepare keyword and semantic search terms with an LLM when available."""

    def __init__(self, llm_provider: OpenAIProvider | None = None):
        self.llm_provider = llm_provider

    def __call__(self, state: QAState) -> QAState:
        """Input: clean question and topic hint. Output: search query, terms, filters."""

        def work():
            base_question = state.get("clean_question", "")
            topic_hint = state.get("topic_hint")
            errors = list(state.get("errors", []))
            plan_notes: list[str] = []
            planner_used = "fallback"
            try:
                if self.llm_provider is None or not hasattr(self.llm_provider, "generate_query_plan"):
                    raise ConfigurationError("LLM query planner is unavailable.")
                plan = self.llm_provider.generate_query_plan(
                    QueryPlanRequest(
                        question=base_question,
                        topic_hint=topic_hint,
                    )
                )
                clean_question = plan.clean_question or base_question
                clarified_question = plan.clarified_question or clean_question
                terms = _normalize_terms(plan.query_terms, clarified_question)
                search_query = plan.search_query or " ".join(terms)
                plan_notes = plan.notes
                planner_used = "llm"
            except Exception as exc:
                clean_question = base_question
                clarified_question = base_question
                terms = expand_query_terms(base_question)
                search_query = " ".join(dict.fromkeys(terms))
                plan_notes = [f"Planner fallback used: {exc}"]
                errors.append(f"Query planner fallback used: {exc}")
            filters = {"topic_hint": state.get("topic_hint")} if state.get("topic_hint") else {}
            return {
                "clean_question": clean_question,
                "clarified_question": clarified_question,
                "search_query": search_query,
                "query_terms": terms,
                "planner_notes": plan_notes,
                "planner_used": planner_used,
                "filters": filters,
                "errors": errors,
            }

        return _timed("QueryPlanner", state, work)


class RetrievalNode:
    """Run both keyword and semantic retrieval through KnowledgeService."""

    def __init__(self, knowledge_service: KnowledgeService):
        self.knowledge_service = knowledge_service

    def __call__(self, state: QAState) -> QAState:
        """Input: search query and filters. Output: keyword/semantic chunks and errors."""

        def work():
            keyword, semantic, semantic_error = self.knowledge_service.search(
                state.get("search_query", ""),
                topic_hint=state.get("topic_hint"),
            )
            errors = list(state.get("errors", []))
            if semantic_error:
                errors.append(semantic_error)
            return {
                "keyword_results": keyword,
                "semantic_results": semantic,
                "semantic_error": semantic_error,
                "errors": errors,
            }

        return _timed("KeywordRetriever+SemanticRetriever", state, work)


class ResultMerger:
    """Merge keyword and semantic chunks with Reciprocal Rank Fusion."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def __call__(self, state: QAState) -> QAState:
        """Input: retrieval result lists. Output: deduplicated ranked chunks."""

        def work():
            scores: dict[str, float] = {}
            chunks_by_id: dict[str, RetrievedChunk] = {}
            for result_list in (state.get("keyword_results", []), state.get("semantic_results", [])):
                for chunk in result_list:
                    scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                        1.0 / (self.settings.hybrid_rrf_k + chunk.rank)
                    )
                    chunks_by_id[chunk.chunk_id] = chunk
            ranked = sorted(chunks_by_id.values(), key=lambda item: scores[item.chunk_id], reverse=True)
            return {"merged_results": ranked[: self.settings.final_context_top_k]}

        return _timed("ResultMerger", state, work)


class NeighborExpander:
    """Add one previous and one next approved chunk around ranked chunks."""

    def __init__(self, knowledge_service: KnowledgeService):
        self.knowledge_service = knowledge_service

    def __call__(self, state: QAState) -> QAState:
        """Input: `merged_results`. Output: `context_results` with neighbors."""

        def work():
            return {
                "context_results": self.knowledge_service.add_neighbor_chunks(
                    state.get("merged_results", []),
                    neighbor_window=1,
                )
            }

        return _timed("NeighborExpander", state, work)


class ContextBuilder:
    """Build compact evidence context for answer generation."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def __call__(self, state: QAState) -> QAState:
        """Input: merged chunks. Output: evidence context string and citations."""

        def work():
            token_total = 0
            parts: list[str] = []
            citations: list[CitationOut] = []
            selected_chunks: list[RetrievedChunk] = []
            for chunk in state.get("context_results", state.get("merged_results", [])):
                if token_total + chunk.token_count > self.settings.max_context_tokens:
                    break
                token_total += chunk.token_count
                selected_chunks.append(chunk)
                page_or_section = _page_or_section(chunk)
                parts.append(
                    f"[{chunk.source_id} / {chunk.chunk_id}] {chunk.title}, {page_or_section}\n"
                    f"Content type: {chunk.content_type}\n"
                    f"{chunk.text}"
                )
                citations.append(
                    CitationOut(
                        source_id=chunk.source_id,
                        chunk_id=chunk.chunk_id,
                        title=chunk.title,
                        page_or_section=page_or_section,
                    )
                )
            return {
                "context_results": selected_chunks,
                "evidence_context": "\n\n---\n\n".join(parts),
                "context_citations": citations,
                "citations": citations,
            }

        return _timed("ContextBuilder", state, work)


class AnswerGenerator:
    """Generate a final draft answer from the compact context."""

    def __init__(self, settings: Settings, llm_provider: OpenAIProvider):
        self.settings = settings
        self.llm_provider = llm_provider

    def __call__(self, state: QAState) -> QAState:
        """Input: question, context, and safety flags. Output: draft answer."""

        def work():
            if not state.get("evidence_context"):
                return {
                    "draft_answer": _insufficient_evidence_answer(),
                    "insufficient_evidence": True,
                }
            try:
                answer = self.llm_provider.generate_answer(
                    AnswerRequest(
                        question=state.get("clean_question", ""),
                        evidence_context=state.get("evidence_context", ""),
                        safety_mode=state.get("safety_mode", "normal"),
                        safety_flags=state.get("safety_flags", []),
                        max_tokens=self.settings.answer_max_tokens,
                        memory_context=state.get("memory_context_rendered"),
                    )
                )
                return {"draft_answer": answer, "insufficient_evidence": False}
            except ConfigurationError as exc:
                return {
                    "draft_answer": _fallback_answer_from_evidence(state, f"Configuration error: {exc}"),
                    "insufficient_evidence": False,
                    "errors": [*state.get("errors", []), str(exc)],
                }
            except Exception as exc:
                return {
                    "draft_answer": _fallback_answer_from_evidence(
                        state,
                        f"Answer generation failed: {exc}",
                    ),
                    "insufficient_evidence": False,
                    "errors": [*state.get("errors", []), f"Answer generation failed: {exc}"],
                }

        return _timed("AnswerGenerator", state, work)


class CitationChecker:
    """Ensure answer citations are valid and match the returned citation list."""

    def __call__(self, state: QAState) -> QAState:
        """Input: draft answer and context citations. Output: checked answer/citations."""

        def work():
            citations: list[CitationOut] = state.get("citations", [])
            context_citations = state.get("context_citations", [])
            if not context_citations:
                context_citations = _citations_from_context_chunks(state.get("context_results", []))
            citations_by_pair = {(citation.source_id, citation.chunk_id): citation for citation in citations}
            for citation in context_citations:
                citations_by_pair.setdefault((citation.source_id, citation.chunk_id), citation)
            valid_pairs = set(citations_by_pair)
            answer = state.get("draft_answer", "")
            cited_pairs_in_answer = list(
                dict.fromkeys(
                    re.findall(r"\[(src_[A-Za-z0-9]+)\s*/\s*(chk_[A-Za-z0-9]+)\]", answer)
                )
            )
            cited_pairs = set(cited_pairs_in_answer)
            invalid_pairs = cited_pairs - valid_pairs
            errors = list(state.get("errors", []))
            if invalid_pairs:
                repaired_pairs: dict[tuple[str, str], CitationOut] = {}
                removed_pairs: set[tuple[str, str]] = set()

                def scrub(match: re.Match) -> str:
                    pair = (match.group(1), match.group(2))
                    if pair not in invalid_pairs:
                        return match.group(0)
                    replacement = _find_context_citation_for_invalid_marker(
                        answer=answer,
                        match=match,
                        invalid_pair=pair,
                        context_citations=context_citations,
                    )
                    if replacement:
                        repaired_pairs[pair] = replacement
                        citations_by_pair[(replacement.source_id, replacement.chunk_id)] = replacement
                        return f"[{replacement.source_id} / {replacement.chunk_id}]"
                    removed_pairs.add(pair)
                    return match.group(0)

                answer = re.sub(
                    r"\[(src_[A-Za-z0-9]+)\s*/\s*(chk_[A-Za-z0-9]+)\]",
                    scrub,
                    answer,
                )
                if repaired_pairs:
                    errors.append(
                        f"Unsupported citation(s) repaired from final context: {sorted(repaired_pairs)}"
                    )
                if removed_pairs:
                    errors.append(f"Unsupported citation(s) removed: {sorted(removed_pairs)}")

                    def remove_unrepaired(match: re.Match) -> str:
                        pair = (match.group(1), match.group(2))
                        if pair in removed_pairs:
                            return "[unsupported citation removed]"
                        return match.group(0)

                    answer = re.sub(
                        r"\[(src_[A-Za-z0-9]+)\s*/\s*(chk_[A-Za-z0-9]+)\]",
                        remove_unrepaired,
                        answer,
                    )
                    answer += "\n\nCitation check note: unsupported citation markers were removed."
                cited_pairs_in_answer = list(
                    dict.fromkeys(
                        re.findall(r"\[(src_[A-Za-z0-9]+)\s*/\s*(chk_[A-Za-z0-9]+)\]", answer)
                    )
                )
                cited_pairs = set(cited_pairs_in_answer)

            filtered_citations = [
                citations_by_pair[pair]
                for pair in cited_pairs_in_answer
                if pair in citations_by_pair
            ]
            if "- None" in answer and not cited_pairs:
                filtered_citations = []
            answer = _normalize_numeric_citation_punctuation(answer)

            return {
                "final_answer": answer,
                "citations": filtered_citations,
                "insufficient_evidence": state.get("insufficient_evidence", False)
                or _answer_says_insufficient(answer),
                "errors": errors,
            }

        return _timed("CitationChecker", state, work)


def _citations_from_context_chunks(chunks: list[RetrievedChunk]) -> list[CitationOut]:
    """Build citation records for chunks that were actually sent to the LLM."""

    return [
        CitationOut(
            source_id=chunk.source_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            page_or_section=_page_or_section(chunk),
        )
        for chunk in chunks
    ]


def _find_context_citation_for_invalid_marker(
    answer: str,
    match: re.Match,
    invalid_pair: tuple[str, str],
    context_citations: list[CitationOut],
) -> CitationOut | None:
    """Find a final-context citation that matches the invalid marker's line."""

    line_tail = _line_tail_after_match(answer, match)
    candidates = [
        citation
        for citation in context_citations
        if citation.source_id == invalid_pair[0] and _citation_metadata_matches(line_tail, citation)
    ]
    if len(candidates) == 1:
        return candidates[0]
    candidates = [
        citation
        for citation in context_citations
        if _citation_metadata_matches(line_tail, citation)
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _line_tail_after_match(answer: str, match: re.Match) -> str:
    """Return text after a marker on the same answer line."""

    line_end = answer.find("\n", match.end())
    if line_end == -1:
        line_end = len(answer)
    return answer[match.end() : line_end]


def _citation_metadata_matches(text: str, citation: CitationOut) -> bool:
    """Match answer-line citation metadata to one final-context citation."""

    normalized_text = _normalize_citation_text(text)
    normalized_title = _normalize_citation_text(citation.title)
    title_matches = bool(normalized_title and normalized_title in normalized_text)
    location_matches = _citation_location_matches(text, citation.page_or_section)
    if citation.page_or_section and "page " in citation.page_or_section:
        return location_matches and (title_matches or not normalized_title)
    if citation.page_or_section:
        normalized_location = _normalize_citation_text(citation.page_or_section)
        return title_matches and normalized_location in normalized_text
    return title_matches


def _citation_location_matches(text: str, page_or_section: str | None) -> bool:
    """Return true when answer-line text mentions the citation page/section."""

    if not page_or_section:
        return False
    page_match = re.search(r"page\s+(\d+)", page_or_section, flags=re.IGNORECASE)
    if not page_match:
        return _normalize_citation_text(page_or_section) in _normalize_citation_text(text)
    citation_page = int(page_match.group(1))
    for start_text, end_text in re.findall(r"page\s+(\d+)(?:\s*[-–]\s*(\d+))?", text, flags=re.IGNORECASE):
        start_page = int(start_text)
        end_page = int(end_text) if end_text else start_page
        if start_page <= citation_page <= end_page:
            return True
    return False


def _normalize_citation_text(text: str) -> str:
    """Normalize citation titles and locations for conservative matching."""

    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z]+", " ", text).lower()).strip()


def _normalize_terms(query_terms: list[str], clarified_question: str) -> list[str]:
    """Deduplicate LLM query terms while preserving the clarified question first."""

    ordered = [clarified_question, *query_terms]
    return list(dict.fromkeys(term.strip() for term in ordered if term and term.strip()))[:12]


def _page_or_section(chunk: RetrievedChunk) -> str:
    if chunk.page_number:
        return f"page {chunk.page_number}"
    if chunk.section_heading:
        return chunk.section_heading
    return "section unavailable"


def _insufficient_evidence_answer() -> str:
    return (
        "Short answer:\n"
        "I do not have enough approved evidence in the knowledge base to answer this reliably.\n\n"
        "Practical guidance:\n"
        "Upload and approve a relevant source, then ask again.\n\n"
        "Evidence used:\n"
        "- None\n\n"
        "Safety note:\n"
        "No evidence-grounded recommendation was generated."
    )


def _answer_says_insufficient(answer: str) -> bool:
    """Return true when the model explicitly refuses because evidence is insufficient."""

    normalized = answer.lower()
    return "insufficient evidence" in normalized or "not enough approved evidence" in normalized


def _normalize_numeric_citation_punctuation(answer: str) -> str:
    """Move inline numeric citation markers before terminal punctuation."""

    return re.sub(r"([.!?])\s*((?:\[\d+(?:\s*,\s*\d+)*\])+)", r" \2\1", answer)


def _fallback_answer_from_evidence(state: QAState, provider_error: str) -> str:
    """Build a conservative answer when the chat model provider fails.

    Parameters:
        state: QAState containing `clean_question`, `citations`, and retrieved context.
        provider_error: short explanation of the failed model-provider call.
    Returns:
        str: answer in the standard answer format, grounded only in retrieved snippets.
    """

    citations: list[CitationOut] = state.get("citations", [])
    context_chunks: list[RetrievedChunk] = state.get("context_results", state.get("merged_results", []))
    evidence_lines = _format_evidence_lines(citations)
    snippet_lines = _format_retrieved_snippets(context_chunks)
    question = state.get("clean_question", "the question")
    return (
        "Short answer:\n"
        "I found approved evidence, but the chat model provider failed before it could write a full answer. "
        f"For the question '{question}', use the retrieved evidence below as the current grounded result.\n\n"
        "Practical guidance:\n"
        "Do not treat this fallback as a complete coaching recommendation. Review the evidence snippets, "
        "then fix the chat model API key, model name, quota, or network issue for a full generated answer.\n\n"
        "Retrieved evidence summary:\n"
        f"{snippet_lines}\n\n"
        "Evidence used:\n"
        f"{evidence_lines}\n\n"
        "Safety note:\n"
        "This fallback is evidence display, not personalized medical, nutrition, or training prescription.\n\n"
        f"Provider note:\n{provider_error}"
    )


def _format_evidence_lines(citations: list[CitationOut]) -> str:
    """Format citations in the required `[source_id / chunk_id]` style."""

    if not citations:
        return "- None"
    return "\n".join(
        f"- [{citation.source_id} / {citation.chunk_id}] {citation.title}, "
        f"{citation.page_or_section or 'section unavailable'}"
        for citation in citations
    )


def _format_retrieved_snippets(chunks: list[RetrievedChunk]) -> str:
    """Return short snippets from retrieved chunks for the fallback answer."""

    if not chunks:
        return "- No snippets available."
    lines = []
    for chunk in chunks[:3]:
        snippet = re.sub(r"\s+", " ", chunk.text).strip()
        if len(snippet) > 300:
            snippet = f"{snippet[:297]}..."
        lines.append(f"- [{chunk.source_id} / {chunk.chunk_id}] {snippet}")
    return "\n".join(lines)
