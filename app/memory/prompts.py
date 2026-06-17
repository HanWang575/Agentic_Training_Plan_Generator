"""Prompt builders for LLM-backed memory nodes."""

import json
from typing import Any


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def build_memory_plan_router_prompt(
    latest_user_query: str,
    task_type: str,
    topic_hint: str | None,
    conversation_summary: str,
    active_plan_exists: bool,
) -> str:
    """Build a JSON-only prompt for active-plan context routing."""

    return (
        "You are the PlanContextRouter for a personal trainer memory system.\n"
        "Decide how much active training-plan context should be loaded for this request.\n\n"
        "Allowed levels:\n"
        "- none: the request is unrelated to the current plan.\n"
        "- summary: the request is training-related but does not need full plan details.\n"
        "- full: the request asks about the current plan, today's workout, planned sessions, missed/completed "
        "sessions, adaptation, progression, schedule changes, fatigue affecting the plan, or plan review/update.\n\n"
        "Rules:\n"
        "- If no active plan exists, return none.\n"
        "- Calls from program_generator or plan_review should return full when an active plan exists.\n"
        "- When uncertain between none and summary, choose summary.\n"
        "- Return JSON only.\n\n"
        f"Active plan exists: {active_plan_exists}\n"
        f"Task type: {task_type}\n"
        f"Topic hint: {topic_hint or ''}\n"
        f"Latest user query:\n{latest_user_query}\n\n"
        f"Conversation summary:\n{conversation_summary or ''}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "plan_context_level": "none | summary | full",\n'
        '  "rationale": "short reason",\n'
        '  "confidence": 0.0\n'
        "}"
    )


def build_rolling_summary_prompt(existing_summary: str, messages: list[dict[str, Any]]) -> str:
    """Build a JSON-only prompt for rolling conversation summaries."""

    return (
        "You are the RollingSummaryUpdater for a personal trainer memory system.\n"
        "Compact older conversation turns into a durable working-memory summary.\n\n"
        "Rules:\n"
        "- Preserve unresolved user commitments, pending questions, constraints, goals, injuries, preferences, "
        "recent readiness issues, and important assistant commitments.\n"
        "- Do not invent facts.\n"
        "- Keep the summary concise and readable.\n"
        "- Return JSON only.\n\n"
        f"Existing rolling summary:\n{existing_summary or ''}\n\n"
        f"Older messages to summarize:\n{_json(messages)}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "rolling_summary": "updated compact summary"\n'
        "}"
    )


def build_rolling_summary_compression_prompt(rolling_summary: str, target_tokens: int) -> str:
    """Build a JSON-only prompt for compressing an oversized rolling summary."""

    return (
        "You are the RollingSummaryCompressor for a personal trainer memory system.\n"
        "Rewrite an oversized rolling conversation summary so it fits the target budget.\n\n"
        "Rules:\n"
        "- Preserve unresolved user commitments, pending questions, durable constraints, goals, injuries, "
        "preferences, recent readiness issues, and important assistant commitments.\n"
        "- Prefer current and unresolved facts over stale details.\n"
        "- Do not invent facts.\n"
        "- Keep the summary concise and readable.\n"
        "- Return JSON only.\n\n"
        f"Target budget: about {target_tokens} tokens.\n\n"
        f"Oversized rolling summary:\n{rolling_summary or ''}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "rolling_summary": "rewritten compact summary"\n'
        "}"
    )


def build_episode_summary_prompt(
    title: str | None,
    topic: str | None,
    rolling_summary: str,
    messages: list[dict[str, Any]],
) -> str:
    """Build a JSON-only prompt for episodic memory summarization."""

    return (
        "You are the EpisodeSummarizer for a personal trainer memory system.\n"
        "Convert a closed conversation into a permanent episodic memory.\n\n"
        "Rules:\n"
        "- Summarize only what appears in the conversation or rolling summary.\n"
        "- Preserve actionable user context, decisions, stated goals, constraints, injuries/pain, readiness, "
        "preferences, and follow-up needs.\n"
        "- Topics should be short lowercase labels such as strength, nutrition, recovery, injury, running, plan, "
        "preference, or general.\n"
        "- Importance is 0.5 normally, 0.7 for goals/plans/performance metrics, and 0.85 for safety-relevant "
        "health, pain, injury, illness, or allergy context.\n"
        "- Return JSON only.\n\n"
        f"Conversation title: {title or ''}\n"
        f"Conversation topic: {topic or ''}\n"
        f"Rolling summary:\n{rolling_summary or ''}\n\n"
        f"Messages:\n{_json(messages)}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "summary": "episode summary",\n'
        '  "topics": ["topic"],\n'
        '  "importance": 0.5\n'
        "}"
    )


def build_memory_extractor_prompt(
    message: dict[str, Any],
    profile_keys: list[str],
    recent_status_keys: list[str],
) -> str:
    """Build a JSON-only prompt for candidate memory extraction."""

    return (
        "You are the MemoryExtractor for a personal trainer AI.\n"
        "Extract durable or recent user facts from one user message as reviewable memory candidates.\n\n"
        "Extract only user-provided facts that could help future exercise, recovery, nutrition, relaxation, "
        "safety, or planning. Do not extract assistant advice, generic science claims, or guesses.\n\n"
        "Target types:\n"
        "- ProfileField: stable or long-term facts such as goals, preferences, constraints, equipment, PRs, "
        "training history, body metrics, allergies, and durable dislikes.\n"
        "- RecentStatus: short-lived readiness/status facts such as sleep, soreness, pain, illness, stress, "
        "schedule pressure, available time, recent adherence, or temporary limits.\n\n"
        "Rules:\n"
        "- Prefer one of the approved keys when it fits.\n"
        "- If no approved key fits but the fact is useful, propose a concise snake_case key.\n"
        "- Mark safety_relevant true for pain, injury, illness, medication, allergy, eating-disorder, or other "
        "health/safety-sensitive facts.\n"
        "- Confidence should be 0.5 to 1.0.\n"
        "- Return JSON only.\n\n"
        f"Approved ProfileField keys:\n{_json(profile_keys)}\n\n"
        f"Approved RecentStatus keys:\n{_json(recent_status_keys)}\n\n"
        f"Message:\n{_json(message)}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "category": "profile | recent_status | preference | constraint | performance | goal",\n'
        '      "target_type": "ProfileField | RecentStatus",\n'
        '      "memory_key": "snake_case_key",\n'
        '      "text": "short human-readable memory sentence",\n'
        '      "value": "string, number, boolean, object, or array",\n'
        '      "confidence": 0.7,\n'
        '      "safety_relevant": false\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def build_memory_key_proposal_prompt(
    target_type: str,
    proposed_key: str,
    example_value: Any,
    reason: str,
    existing_keys: list[str],
) -> str:
    """Build a JSON-only prompt for proposed memory-key equivalence review."""

    return (
        "You are the MemoryKeyProposalDetector for a personal trainer memory system.\n"
        "Decide whether a proposed memory key is meaningfully new or equivalent to an existing approved key.\n\n"
        "Rules:\n"
        "- duplicate means it is effectively the same key.\n"
        "- equivalent means an existing key should be used instead even though wording differs.\n"
        "- not_duplicate means the proposed key captures useful information not covered by existing keys.\n"
        "- Return JSON only.\n\n"
        f"Target type: {target_type}\n"
        f"Proposed key: {proposed_key}\n"
        f"Example value:\n{_json(example_value)}\n"
        f"Reason:\n{reason}\n"
        f"Existing approved keys:\n{_json(existing_keys)}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "duplicate_decision": "not_duplicate | equivalent | duplicate",\n'
        '  "nearest_existing_keys": ["existing_key"],\n'
        '  "canonical_key": "existing_key_or_empty",\n'
        '  "reason": "short reason"\n'
        "}"
    )


def build_completed_plan_episode_prompt(plan_payload: dict[str, Any]) -> str:
    """Build a JSON-only prompt for completed-plan episodic summaries."""

    return (
        "You are the CompletedPlanEpisodeWriter for a personal trainer memory system.\n"
        "Summarize a completed training plan as an episodic memory for future planning reflection.\n\n"
        "Rules:\n"
        "- Preserve plan ID/version, final status, useful progress patterns, completion/missed/adapted sessions, "
        "important feedback, constraints, and lessons for future planning.\n"
        "- Do not invent outcomes that are not in the payload.\n"
        "- Topics should include training_plan and plan_completion plus any useful short labels.\n"
        "- Return JSON only.\n\n"
        f"Completed plan payload:\n{_json(plan_payload)}\n\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "summary": "completed plan summary",\n'
        '  "topics": ["training_plan", "plan_completion"],\n'
        '  "importance": 0.75\n'
        "}"
    )
