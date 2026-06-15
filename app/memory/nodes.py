"""Small deterministic memory workflow nodes."""

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

    def __call__(self, state: MemoryWorkflowState) -> MemoryWorkflowState:
        """Input: latest query and active-plan flag. Output: plan context level."""

        query = state.get("latest_user_query", "").lower()
        active_plan_exists = bool(state.get("active_plan_progress"))
        if not active_plan_exists:
            return {"plan_context_level": "none"}
        if state.get("task_type") in {"program_generator", "plan_review"}:
            return {"plan_context_level": "full"}
        if any(term in query for term in FULL_PLAN_TERMS):
            return {"plan_context_level": "full"}
        if any(term in query for term in SUMMARY_PLAN_TERMS):
            return {"plan_context_level": "summary"}
        return {"plan_context_level": "none"}


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

    def __init__(self, max_tokens: int = 34000, section_caps: dict[str, int] | None = None):
        self.max_tokens = max_tokens
        self.section_caps = section_caps or dict(self.DEFAULT_SECTION_CAPS)

    def __call__(self, state: MemoryWorkflowState) -> MemoryWorkflowState:
        """Input: memory_context sections. Output: compacted memory_context."""

        memory_context = state.get("memory_context")
        if memory_context is None:
            return {}
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

        memory_context.sections = sections
        memory_context.rendered_context = render_sections(sections)
        memory_context.token_estimate = _section_token_total(sections)
        memory_context.compaction_report = report
        return {"memory_context": memory_context, "compaction_report": report}


def render_sections(sections: dict[str, str]) -> str:
    """Render memory context sections into a compact prompt payload."""

    parts = []
    for name, text in sections.items():
        if text.strip():
            parts.append(f"## {name.replace('_', ' ').title()}\n{text.strip()}")
    return "\n\n".join(parts)


def _section_token_total(sections: dict[str, str]) -> int:
    return sum(estimate_tokens(text) for text in sections.values())

