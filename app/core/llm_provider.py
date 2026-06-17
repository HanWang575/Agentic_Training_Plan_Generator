"""Provider adapter used by embedding and answer-generation nodes."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import json
import re

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.memory.prompts import (
    build_completed_plan_episode_prompt,
    build_episode_summary_prompt,
    build_memory_extractor_prompt,
    build_memory_key_proposal_prompt,
    build_memory_plan_router_prompt,
    build_rolling_summary_compression_prompt,
    build_rolling_summary_prompt,
)
from app.qa.prompts import build_answer_prompt, build_query_planner_prompt


@dataclass(slots=True)
class AnswerRequest:
    """Input shape for answer generation."""

    question: str
    evidence_context: str
    safety_mode: str
    safety_flags: list[dict]
    max_tokens: int
    memory_context: str | None = None


@dataclass(slots=True)
class QueryPlanRequest:
    """Input shape for query planning."""

    question: str
    topic_hint: str | None = None


@dataclass(slots=True)
class QueryPlanResult:
    """Output shape for LLM-based query planning."""

    clean_question: str
    clarified_question: str
    search_query: str
    query_terms: list[str]
    notes: list[str]


@dataclass(slots=True)
class MemoryPlanRouteRequest:
    """Input shape for LLM-based active-plan context routing."""

    latest_user_query: str
    task_type: str
    active_plan_exists: bool
    topic_hint: str | None = None
    conversation_summary: str = ""


@dataclass(slots=True)
class MemoryPlanRouteResult:
    """Output shape for LLM-based active-plan context routing."""

    plan_context_level: str
    rationale: str
    confidence: float


@dataclass(slots=True)
class MemoryRollingSummaryRequest:
    """Input shape for LLM rolling-summary updates."""

    existing_summary: str
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class MemoryRollingSummaryCompressionRequest:
    """Input shape for LLM rolling-summary self-compression."""

    rolling_summary: str
    target_tokens: int


@dataclass(slots=True)
class MemoryEpisodeSummaryRequest:
    """Input shape for LLM episodic summaries."""

    title: str | None
    topic: str | None
    rolling_summary: str
    messages: list[dict[str, Any]]


@dataclass(slots=True)
class MemoryEpisodeSummaryResult:
    """Output shape for LLM episodic summaries."""

    summary: str
    topics: list[str]
    importance: float


@dataclass(slots=True)
class MemoryExtractionRequest:
    """Input shape for LLM memory extraction."""

    message: dict[str, Any]
    profile_keys: list[str]
    recent_status_keys: list[str]


@dataclass(slots=True)
class MemoryExtractionResult:
    """Output shape for LLM memory extraction."""

    items: list[dict[str, Any]]


@dataclass(slots=True)
class MemoryKeyProposalRequest:
    """Input shape for LLM key-equivalence review."""

    target_type: str
    proposed_key: str
    example_value: Any
    reason: str
    existing_keys: list[str]


@dataclass(slots=True)
class MemoryKeyProposalResult:
    """Output shape for LLM key-equivalence review."""

    duplicate_decision: str
    nearest_existing_keys: list[str]
    canonical_key: str
    reason: str


@dataclass(slots=True)
class CompletedPlanEpisodeRequest:
    """Input shape for LLM completed-plan episodic summaries."""

    plan_payload: dict[str, Any]


class OpenAIProvider:
    """Small adapter around OpenAI-compatible APIs.

    The rest of the code calls this adapter instead of importing the OpenAI SDK
    directly. That keeps graph nodes easy to test with fake providers.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._chat_client = None
        self._embedding_client = None

    def _require_chat_client(self):
        """Return the chat client or raise a clear configuration error."""

        provider = self.settings.normalized_llm_model_type
        if provider not in {"openai", "aihubmix"}:
            raise ConfigurationError(
                f"Unsupported LLM_MODEL_TYPE '{self.settings.llm_model_type}'. "
                "Supported values are 'openai' and 'aihubmix'."
            )
        api_key = self.settings.active_chat_api_key
        if not api_key:
            key_name = "OPENAI_API_KEY" if provider == "openai" else "LLM_API_KEY"
            raise ConfigurationError(
                f"{key_name} is not configured. The app can start, but answer generation needs a chat API key."
            )
        if self._chat_client is None:
            from openai import OpenAI

            client_kwargs = {"api_key": api_key}
            if provider == "aihubmix":
                if not self.settings.active_chat_base_url:
                    raise ConfigurationError(
                        "LLM_BASE_URL is not configured. aihubmix needs an OpenAI-compatible base URL."
                    )
                client_kwargs["base_url"] = self.settings.active_chat_base_url
            self._chat_client = OpenAI(**client_kwargs)
        return self._chat_client

    def _require_embedding_client(self):
        """Return an OpenAI-compatible embedding client.

        Supported provider values:
            `openai`: uses OPENAI_API_KEY and OpenAI's default base URL.
            `dashscope`: uses EMBED_API_KEY and EMBED_BASE_URL.
        """

        provider = self.settings.normalized_embed_model_type
        if provider not in {"openai", "dashscope"}:
            raise ConfigurationError(
                f"Unsupported EMBED_MODEL_TYPE '{self.settings.embed_model_type}'. "
                "Supported values are 'openai' and 'dashscope'."
            )
        api_key = self.settings.active_embedding_api_key
        if not api_key:
            key_name = "OPENAI_API_KEY" if provider == "openai" else "EMBED_API_KEY"
            raise ConfigurationError(
                f"{key_name} is not configured. Semantic indexing and retrieval need an embedding API key."
            )
        if self._embedding_client is None:
            from openai import OpenAI

            client_kwargs = {"api_key": api_key}
            if provider == "dashscope":
                if not self.settings.active_embedding_base_url:
                    raise ConfigurationError(
                        "EMBED_BASE_URL is not configured. DashScope compatible mode needs a base URL."
                    )
                client_kwargs["base_url"] = self.settings.active_embedding_base_url
            self._embedding_client = OpenAI(**client_kwargs)
        return self._embedding_client

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create embeddings.

        Parameters:
            texts: sequence with shape `[text_count]`.
        Returns:
            list[list[float]]: shape `[text_count, embedding_dimensions]`.
        """

        client = self._require_embedding_client()
        request = {
            "model": self.settings.active_embedding_model,
            "input": list(texts),
        }
        if self.settings.active_embedding_dimensions is not None:
            request["dimensions"] = self.settings.active_embedding_dimensions
        response = client.embeddings.create(**request)
        return [item.embedding for item in response.data]

    def generate_answer(self, request: AnswerRequest) -> str:
        """Generate a grounded answer from the compact evidence context.

        Parameters:
            request: AnswerRequest with the user question and retrieved context.
        Returns:
            str: answer formatted with Short answer, Practical guidance,
            Evidence used, and Safety note sections.
        """

        prompt = build_answer_prompt(
            question=request.question,
            evidence_context=request.evidence_context,
            safety_mode=request.safety_mode,
            safety_flags=request.safety_flags,
            memory_context=request.memory_context,
        )
        return self._generate_chat_text(prompt, max_tokens=request.max_tokens)

    def generate_query_plan(self, request: QueryPlanRequest) -> QueryPlanResult:
        """Generate a retrieval-oriented query plan from the user's question."""

        prompt = build_query_planner_prompt(
            question=request.question,
            topic_hint=request.topic_hint,
        )
        raw_text = self._generate_chat_text(prompt, max_tokens=500)
        payload = _parse_json_object(raw_text)
        return QueryPlanResult(
            clean_question=str(payload.get("clean_question", "")).strip(),
            clarified_question=str(payload.get("clarified_question", "")).strip(),
            search_query=str(payload.get("search_query", "")).strip(),
            query_terms=[
                str(item).strip()
                for item in payload.get("query_terms", [])
                if str(item).strip()
            ],
            notes=[
                str(item).strip()
                for item in payload.get("notes", [])
                if str(item).strip()
            ],
        )

    def generate_memory_plan_route(self, request: MemoryPlanRouteRequest) -> MemoryPlanRouteResult:
        """Generate the active-plan context routing decision for memory context."""

        prompt = build_memory_plan_router_prompt(
            latest_user_query=request.latest_user_query,
            task_type=request.task_type,
            topic_hint=request.topic_hint,
            conversation_summary=request.conversation_summary,
            active_plan_exists=request.active_plan_exists,
        )
        payload = _parse_json_object(self._generate_chat_text(prompt, max_tokens=350))
        level = str(payload.get("plan_context_level", "")).strip().lower()
        if level not in {"none", "summary", "full"}:
            raise ValueError(f"Invalid plan_context_level from LLM: {level}")
        return MemoryPlanRouteResult(
            plan_context_level=level,
            rationale=str(payload.get("rationale", "")).strip(),
            confidence=_clamp_float(payload.get("confidence"), default=0.0),
        )

    def generate_memory_rolling_summary(self, request: MemoryRollingSummaryRequest) -> str:
        """Generate an updated rolling conversation summary."""

        prompt = build_rolling_summary_prompt(
            existing_summary=request.existing_summary,
            messages=request.messages,
        )
        payload = _parse_json_object(self._generate_chat_text(prompt, max_tokens=900))
        summary = str(payload.get("rolling_summary", "")).strip()
        if not summary:
            raise ValueError("RollingSummaryUpdater returned an empty summary.")
        return summary

    def generate_memory_rolling_summary_compression(
        self,
        request: MemoryRollingSummaryCompressionRequest,
    ) -> str:
        """Rewrite an oversized rolling summary to its target budget."""

        prompt = build_rolling_summary_compression_prompt(
            rolling_summary=request.rolling_summary,
            target_tokens=request.target_tokens,
        )
        payload = _parse_json_object(self._generate_chat_text(prompt, max_tokens=max(300, request.target_tokens)))
        summary = str(payload.get("rolling_summary", "")).strip()
        if not summary:
            raise ValueError("RollingSummaryCompressor returned an empty summary.")
        return summary

    def generate_memory_episode_summary(
        self,
        request: MemoryEpisodeSummaryRequest,
    ) -> MemoryEpisodeSummaryResult:
        """Generate a permanent episodic summary for a closed conversation."""

        prompt = build_episode_summary_prompt(
            title=request.title,
            topic=request.topic,
            rolling_summary=request.rolling_summary,
            messages=request.messages,
        )
        payload = _parse_json_object(self._generate_chat_text(prompt, max_tokens=1000))
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise ValueError("EpisodeSummarizer returned an empty summary.")
        return MemoryEpisodeSummaryResult(
            summary=summary,
            topics=_string_list(payload.get("topics", [])) or ["general"],
            importance=_clamp_float(payload.get("importance"), default=0.5),
        )

    def generate_memory_candidates(self, request: MemoryExtractionRequest) -> MemoryExtractionResult:
        """Generate reviewable candidate memory items from one message."""

        prompt = build_memory_extractor_prompt(
            message=request.message,
            profile_keys=request.profile_keys,
            recent_status_keys=request.recent_status_keys,
        )
        payload = _parse_json_object(self._generate_chat_text(prompt, max_tokens=1400))
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise ValueError("MemoryExtractor returned a non-list items field.")
        return MemoryExtractionResult(items=[item for item in items if isinstance(item, dict)])

    def review_memory_key_proposal(self, request: MemoryKeyProposalRequest) -> MemoryKeyProposalResult:
        """Review whether a proposed key duplicates an approved memory key."""

        prompt = build_memory_key_proposal_prompt(
            target_type=request.target_type,
            proposed_key=request.proposed_key,
            example_value=request.example_value,
            reason=request.reason,
            existing_keys=request.existing_keys,
        )
        payload = _parse_json_object(self._generate_chat_text(prompt, max_tokens=500))
        decision = str(payload.get("duplicate_decision", "not_duplicate")).strip().lower()
        if decision not in {"not_duplicate", "equivalent", "duplicate"}:
            decision = "not_duplicate"
        return MemoryKeyProposalResult(
            duplicate_decision=decision,
            nearest_existing_keys=_string_list(payload.get("nearest_existing_keys", []))[:5],
            canonical_key=str(payload.get("canonical_key", "")).strip(),
            reason=str(payload.get("reason", "")).strip(),
        )

    def generate_completed_plan_episode(
        self,
        request: CompletedPlanEpisodeRequest,
    ) -> MemoryEpisodeSummaryResult:
        """Generate an episodic summary for a completed training plan."""

        prompt = build_completed_plan_episode_prompt(request.plan_payload)
        payload = _parse_json_object(self._generate_chat_text(prompt, max_tokens=900))
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise ValueError("CompletedPlanEpisodeWriter returned an empty summary.")
        return MemoryEpisodeSummaryResult(
            summary=summary,
            topics=_string_list(payload.get("topics", [])) or ["training_plan", "plan_completion"],
            importance=_clamp_float(payload.get("importance"), default=0.75),
        )

    def _generate_chat_text(self, prompt: str, max_tokens: int) -> str:
        """Generate text with the configured chat model."""

        client = self._require_chat_client()
        if self.settings.normalized_llm_model_type == "openai" and not self.settings.active_chat_base_url:
            response = client.responses.create(
                model=self.settings.active_chat_model,
                input=prompt,
                max_output_tokens=max_tokens,
            )
            return response.output_text or ""

        response = client.chat.completions.create(
            model=self.settings.active_chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""


def _parse_json_object(text: str) -> dict:
    """Extract one JSON object from a model response."""

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Query planner returned an empty response.")
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return json.loads(cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Query planner did not return a JSON object.")
    return json.loads(match.group(0))


def _string_list(value: Any) -> list[str]:
    """Return a clean string list from a JSON payload value."""

    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _clamp_float(value: Any, default: float) -> float:
    """Parse and clamp a float to the inclusive 0..1 range."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))
