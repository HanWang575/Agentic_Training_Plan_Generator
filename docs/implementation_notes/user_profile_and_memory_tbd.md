# User Profile And Memory TBD Notes

This file records unresolved design and implementation questions for the User Profile And Memory block. It is intentionally separate from the architecture outline and implementation notes.

## 1. Memory Context Budget Strategy

Current concern:

- Final prompt-stage memory context trimming has been removed from the current implementation.
- `MemoryService` now assembles the full `MemoryContextOut` and returns `budget_warnings` when any section or the whole context exceeds its configured token budget.
- Warnings make oversize context visible to the user, but they do not solve the underlying problem of keeping large memory sections concise.

Current mitigations already added:

- Working memory now uses a persisted token-based live window with `rolling_summary_until_message_id`.
- Old working-memory turns are summarized before they leave the live window.
- Pre-summary memory extraction runs before rolling summary updates so candidate memories are not hidden by summarization.

Still TBD:

- Decide whether each memory section needs its own persistent summary/checkpoint strategy.
- Decide how users should resolve repeated section-budget warnings in normal chat usage.
- Decide whether very large profile, recent-status, episode, or training-log sections should have section-specific retrieval/ranking limits before prompt assembly.

## 2. Incomplete Episodic Memory Design

Current concern:

- The system creates `Episode` rows from closed conversations and completed plans.
- `get_memory_context()` can retrieve matching episodes into the `episodic_memories` section.
- The implementation exists, but the product/design purpose of episodes needs to be made sharper.

Key question:

- Why do we need `Episode` and episodic memory context in addition to working memory, profile, recent status, active plan progress, and training logs?

Possible intended roles:

- Preserve conversation-level history after working memory rolls forward.
- Capture useful context that is too broad or narrative to fit profile fields or recent status.
- Support semantic retrieval of past check-ins, decisions, constraints, plan changes, injuries, and user preferences.
- Provide long-term continuity without loading all raw conversation messages.
- Support future planning reflection, for example "what happened last time we tried this plan?"

Still TBD:

- Define which user questions should retrieve episodic memory.
- Define how episodes should be ranked against profile, recent status, and training logs.
- Decide when episodic memory should be trusted for personalization and when it should be treated as stale.
- Decide whether episode summaries need a human review step before entering normal memory context.
- Decide whether episodes should be decomposed into smaller structured facts, kept as narrative summaries, or both.
- Decide how to evaluate whether episodic memory improves answer quality rather than adding noise.
