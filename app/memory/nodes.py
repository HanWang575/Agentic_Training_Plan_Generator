"""Small deterministic memory workflow nodes."""

from app.core.llm_provider import MemoryCompactionRequest, MemoryPlanRouteRequest
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
    """Cheap deterministic token estimate for compaction budgets."""

    if not text:
        return 0
    return max(1, len(text) // 4)


def trim_to_token_budget(text: str, max_tokens: int, keep_newest: bool = False) -> tuple[str, bool]:
    """Trim a section by lines while preserving readable boundaries."""

    if estimate_tokens(text) <= max_tokens:
        return text, False
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return text[: max_tokens * 4], True

    selected: list[str] = []
    total = 0
    iterable = reversed(lines) if keep_newest else lines
    for line in iterable:
        cost = estimate_tokens(line)
        if selected and total + cost > max_tokens:
            break
        selected.append(line)
        total += cost
    if keep_newest:
        selected.reverse()
    return "\n".join(selected), True


class MemoryContextCompactor:
    """Deterministically enforce per-section and global context budgets."""

    DEFAULT_SECTION_CAPS = {
        "working_memory": 6000,
        "rolling_summary": 2000,
        "episodic_memories": 6000,
        "profile": 3000,
        "recent_status": 3000,
        "active_plan_progress": 10000,
        "training_log_summary": 4000,
    }

    GLOBAL_COMPACTION_ORDER = [
        "training_log_summary",
        "episodic_memories",
        "working_memory",
        "rolling_summary",
        "profile",
        "recent_status",
        "active_plan_progress",
    ]

    def __init__(
        self,
        max_tokens: int = 34000,
        section_caps: dict[str, int] | None = None,
        llm_provider=None,
    ):
        self.max_tokens = max_tokens
        self.section_caps = section_caps or dict(self.DEFAULT_SECTION_CAPS)
        self.llm_provider = llm_provider

    def __call__(self, state: MemoryWorkflowState) -> MemoryWorkflowState:
        """Input: memory_context sections. Output: compacted memory_context."""

        memory_context = state.get("memory_context")
        if memory_context is None:
            return {}
        original_sections = dict(memory_context.sections)
        sections = dict(memory_context.sections)
        report: dict[str, dict] = {}
        for section_name, cap in self.section_caps.items():
            if section_name not in sections:
                continue
            compacted, changed = trim_to_token_budget(
                sections[section_name],
                cap,
                keep_newest=section_name == "working_memory",
            )
            if changed:
                report[section_name] = {
                    "reason": "section_cap",
                    "before_tokens": estimate_tokens(sections[section_name]),
                    "after_tokens": estimate_tokens(compacted),
                }
                sections[section_name] = compacted

        total = _section_token_total(sections)
        for section_name in self.GLOBAL_COMPACTION_ORDER:
            if total <= self.max_tokens:
                break
            if not sections.get(section_name):
                continue
            target = max(100, estimate_tokens(sections[section_name]) // 2)
            compacted, changed = trim_to_token_budget(
                sections[section_name],
                target,
                keep_newest=section_name == "working_memory",
            )
            if changed:
                report[section_name] = {
                    "reason": "global_cap",
                    "before_tokens": estimate_tokens(sections[section_name]),
                    "after_tokens": estimate_tokens(compacted),
                }
                sections[section_name] = compacted
                total = _section_token_total(sections)

        errors = list(state.get("errors", []))
        if report and self.llm_provider and hasattr(self.llm_provider, "compact_memory_context"):
            try:
                result = self.llm_provider.compact_memory_context(
                    MemoryCompactionRequest(
                        sections=original_sections,
                        section_caps=self.section_caps,
                        max_tokens=self.max_tokens,
                    )
                )
                candidate_sections = {
                    name: result.sections.get(name, sections.get(name, ""))
                    for name in original_sections
                }
                candidate_sections, llm_trim_report = _enforce_compacted_section_limits(
                    candidate_sections,
                    section_caps=self.section_caps,
                    max_tokens=self.max_tokens,
                )
                sections = candidate_sections
                report = {
                    **result.compaction_report,
                    **{
                        name: {"reason": "llm_post_trim", **details}
                        for name, details in llm_trim_report.items()
                    },
                }
                report.setdefault("_strategy", {"reason": "llm_compaction"})
            except Exception as exc:
                errors.append(f"MemoryContextCompactor LLM fallback used: {exc}")

        memory_context.sections = sections
        memory_context.rendered_context = render_sections(sections)
        memory_context.token_estimate = _section_token_total(sections)
        memory_context.compaction_report = report
        return {"memory_context": memory_context, "compaction_report": report, "errors": errors}


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


def _enforce_compacted_section_limits(
    sections: dict[str, str],
    section_caps: dict[str, int],
    max_tokens: int,
) -> tuple[dict[str, str], dict[str, dict]]:
    report: dict[str, dict] = {}
    cleaned = dict(sections)
    for section_name, cap in section_caps.items():
        if section_name not in cleaned:
            continue
        compacted, changed = trim_to_token_budget(
            cleaned[section_name],
            cap,
            keep_newest=section_name == "working_memory",
        )
        if changed:
            report[section_name] = {
                "before_tokens": estimate_tokens(cleaned[section_name]),
                "after_tokens": estimate_tokens(compacted),
            }
            cleaned[section_name] = compacted

    total = _section_token_total(cleaned)
    for section_name in MemoryContextCompactor.GLOBAL_COMPACTION_ORDER:
        if total <= max_tokens:
            break
        if not cleaned.get(section_name):
            continue
        target = max(100, estimate_tokens(cleaned[section_name]) // 2)
        compacted, changed = trim_to_token_budget(
            cleaned[section_name],
            target,
            keep_newest=section_name == "working_memory",
        )
        if changed:
            report[section_name] = {
                "before_tokens": estimate_tokens(cleaned[section_name]),
                "after_tokens": estimate_tokens(compacted),
            }
            cleaned[section_name] = compacted
            total = _section_token_total(cleaned)
    return cleaned, report
