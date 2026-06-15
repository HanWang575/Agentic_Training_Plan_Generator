# User Profile And Memory Implementation Notes

This file records implementation assumptions and instruction gaps without changing the original architecture or function outline.

## Implemented Scope

- Added SQLModel tables for users, conversations, conversation messages, episodes, candidate memory items, profile fields, recent statuses, key proposals, training plan progress, training log imports, training sessions, and memory traces.
- Added a new `app/memory/` package with approved v1 key catalogs, typed schemas, deterministic LangGraph-compatible nodes, and a `MemoryService` service boundary.
- Added FastAPI memory routes under `/api/memory`.
- Added manual profile editing with approved key validation and explicit extension-mode key proposals.
- Added manual recent-status editing with v1 default `valid_until` estimates.
- Added working memory storage through append-message and conversation-context APIs.
- Added deterministic rolling-summary compaction for long conversations.
- Added idempotent conversation close behavior that writes one episode per conversation.
- Added deterministic candidate extraction for common v1 facts such as goals, lift PRs, sleep, pain, injury, illness, soreness, stress, preferences, dislikes, allergies, height, bodyweight, and training-day availability.
- Added review APIs for approving, rejecting, and expiring candidate memory items.
- Added key-proposal review APIs.
- Added recent-status expiry checks and editable `valid_until` updates.
- Added active training-plan progress storage with append-only JSON event history.
- Added completed-plan episode writing when a tracked plan is marked completed.
- Added Xunji training-log import with idempotence by external ID or checksum and simple normalized training sessions.
- Added memory-context retrieval with profile, recent status, episodic memory, working memory, conditional active-plan context, and imported training-log summary sections.
- Profile and recent-status context now keeps only the latest value per key so downstream LLM nodes do not see conflicting duplicate key/value entries.
- Added deterministic context compaction with section caps and global `MEMORY_CONTEXT_MAX_TOKENS` behavior through settings.
- Added `GET /api/memory/context` so the UI and callers can request full compact memory context by `user_id` with optional `conversation_id`.
- Added `GET /api/memory/users` so the frontend can show existing local user IDs.
- Added `GET /api/memory/conversations` so the frontend can show existing local conversations.
- Added `GET /api/memory/conversations/{conversation_id}` so the frontend can inspect one conversation with all saved messages.
- Added caller-supplied conversation IDs for memory conversation creation.
- Added explicit delete APIs for user-owned memory rows, conversations, profile fields, and recent-status rows.
- Added profile-key, recent-status-key, and current-recent-status endpoints for frontend inspection.
- Added MCP tools for memory context, episodes, memory items, key proposals, recent status, expiring status, active plan progress, and training-log summary.
- Added a basic Streamlit Memory tab and memory-enabled Q&A controls in the Chat tab. The Memory tab keeps full context retrieval separate from episode search, can inspect saved conversation content, and includes filtered review queues for memory items and key proposals.
- Streamlit ID input fields accept IDs pasted either raw or with surrounding JSON/string quotes, such as `"pf_123"`.
- Added focused memory tests and a Q&A integration test.

## QA Block Changes

The Q&A block was intentionally changed only at the optional memory boundary:

- `AskQuestionRequest` now accepts optional `user_id`, `conversation_id`, `include_memory`, and `memory_debug`.
- `QAService.ask()` accepts the same optional fields.
- When `include_memory=true`, `QAService` asks `MemoryService` for a compact context and passes it to answer generation.
- Q&A retrieval from approved evidence remains separate from user memory retrieval.
- The answer prompt labels memory as "personalization only" and says not to treat memory as evidence for scientific claims.
- If `user_id` and `conversation_id` are supplied, `QAService` ensures the memory conversation exists, then best-effort logs the user question and assistant answer through `MemoryService.append_message()`.
- `AnswerTrace` now stores optional `user_id`, `conversation_id`, `memory_context_json`, and `memory_debug_ids_json`.
- Q&A debug output includes memory context and memory IDs when `debug` or `memory_debug` is enabled.
- `README_QA.md` was updated only where the FastAPI route-count smoke test changed because memory routes are now mounted in the same app.

## Assumptions And Gaps

- SQL tables remain the source of truth. Chroma is attempted only as an optional episode-search index when embeddings and Chroma are configured; SQLite keyword search remains the fallback.
- The first memory extractor is deterministic and conservative so the feature works without API keys. It does not replace the future LLM structured extractor described in the outline.
- The PlanContextRouter uses deterministic keyword guardrails in v1. The outline asks for an LLM router; that can be added behind the same service boundary later.
- Recent-status expiry reminders are exposed through API and MCP. No background scheduler has been added yet.
- Conflict handling for manual profile edits updates the current approved field directly because the edit is already user-provided. LLM-extracted candidates still require review before they affect profile or status context.
- Plan progress is tracked, but this block does not generate training plans.
- Training-log import is intentionally provider-tolerant because the exact Xunji payload schema was not provided. The raw payload is stored for audit, and common session fields are normalized when present.
- Memory compaction is deterministic only. No LLM fallback compactor has been added yet.
- The Streamlit Memory tab is a developer/debug UI, not a polished final user experience.
- User-supplied conversation IDs are accepted for local debugging and Chat-tab continuity. Reusing an existing conversation ID for a different user is rejected to avoid cross-user working-memory mixing.
- Destructive debug controls are intentionally ID-based. Deleting a user removes user-owned memory-block rows; deleting a conversation removes its conversation messages, episode, memory traces, and linked memory items. Approved profile and recent-status rows are deleted only through their own row IDs or through user deletion.

## Verification

- Syntax check passed:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app tests
```

- Focused memory and Q&A integration tests passed:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_memory_service.py tests\test_qa_memory_integration.py -q
```

Expected result:

```text
13 passed
```

- FastAPI app import smoke test passed:

```powershell
.\.venv\Scripts\python.exe -c "from app.api.main import app; print(app.title); print(len(app.routes))"
```

Expected result:

```text
Personal Coach Evidence-Based Q&A Assistant
55
```

- Full test suite passed:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result:

```text
69 passed
```
