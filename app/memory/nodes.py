"""Small deterministic memory workflow nodes."""

from typing import Any

from app.core.llm_provider import MemoryPlanRouteRequest
from app.memory.schemas import MemoryWorkflowState


FULL_PLAN_TERMS = {
    "current plan",
    "today's workout",
    "today workout",
    "planned session",
    "missed session",
    "complete session",
    "completed session",
    "adapt",
    "progression",
    "plan review",
    "update my plan",
    "change my plan",
    "next session",
}

SUMMARY_PLAN_TERMS = {
    "training",
    "workout",
    "exercise",
    "recover",
    "recovery",
    "fatigue",
    "soreness",
    "schedule",
    "availability",
    "readiness",
}


MEMORY_SECTION_TOKEN_BUDGETS = {
    "working_memory": 6000,
    "rolling_summary": 2000,
    "episodic_memories": 6000,
    "profile": 3000,
    "recent_status": 3000,
    "active_plan_progress": 5000,
}


class PlanContextRouter:
    """Route a request to none, summary, or full active-plan context."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider

    def __call__(self, state: MemoryWorkflowState) -> MemoryWorkflowState:
        """Input: latest query and active-plan flag. Output: plan context level."""

        active_plan_exists = bool(state.get("active_plan_progress"))
        if not active_plan_exists:
            return {"plan_context_level": "none"}
        errors = list(state.get("errors", []))
        if self.llm_provider and hasattr(self.llm_provider, "generate_memory_plan_route"):
            try:
                result = self.llm_provider.generate_memory_plan_route(
                    MemoryPlanRouteRequest(
                        latest_user_query=state.get("latest_user_query", ""),
                        task_type=state.get("task_type", "qa"),
                        topic_hint=state.get("topic_hint"),
                        conversation_summary=state.get("rolling_summary", ""),
                        active_plan_exists=active_plan_exists,
                    )
                )
                return {
                    "plan_context_level": result.plan_context_level,
                    "plan_context_rationale": result.rationale,
                    "plan_context_confidence": result.confidence,
                    "plan_context_router_used": "llm",
                    "errors": errors,
                }
            except Exception as exc:
                errors.append(f"PlanContextRouter LLM fallback used: {exc}")
        return {
            "plan_context_level": _deterministic_plan_context_level(
                query=state.get("latest_user_query", ""),
                task_type=state.get("task_type", "qa"),
            ),
            "plan_context_router_used": "fallback",
            "errors": errors,
        }


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate for memory-context budgets."""

    if not text:
        return 0
    return max(1, len(text) // 4)


def build_memory_budget_warnings(
    sections: dict[str, str],
    section_caps: dict[str, int],
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Return user-visible warnings without changing the assembled memory context."""

    warnings: list[dict[str, Any]] = []
    for section_name, cap in section_caps.items():
        if section_name not in sections:
            continue
        token_estimate = estimate_tokens(sections[section_name])
        if token_estimate > cap:
            warnings.append(
                {
                    "type": "section_budget_exceeded",
                    "section": section_name,
                    "token_estimate": token_estimate,
                    "token_budget": cap,
                    "over_by": token_estimate - cap,
                    "message": (
                        f"Memory section '{section_name}' is about {token_estimate} tokens, "
                        f"above its {cap}-token budget. The section was not compacted."
                    ),
                }
            )

    total = _section_token_total(sections)
    if total > max_tokens:
        warnings.append(
            {
                "type": "global_budget_exceeded",
                "section": None,
                "token_estimate": total,
                "token_budget": max_tokens,
                "over_by": total - max_tokens,
                "message": (
                    f"Memory context is about {total} tokens, above its {max_tokens}-token budget. "
                    "The context was not compacted."
                ),
            }
        )
    return warnings


def render_sections(sections: dict[str, str]) -> str:
    """Render memory context sections into a compact prompt payload."""

    parts = []
    for name, text in sections.items():
        if text.strip():
            parts.append(f"## {name.replace('_', ' ').title()}\n{text.strip()}")
    return "\n\n".join(parts)


def _section_token_total(sections: dict[str, str]) -> int:
    return sum(estimate_tokens(text) for text in sections.values())


def _deterministic_plan_context_level(query: str, task_type: str) -> str:
    query = query.lower()
    if task_type in {"program_generator", "plan_review"}:
        return "full"
    if any(term in query for term in FULL_PLAN_TERMS):
        return "full"
    if any(term in query for term in SUMMARY_PLAN_TERMS):
        return "summary"
    return "none"
