# User Profile And Memory Implementation Notes

This file records implementation assumptions and instruction gaps without changing the original architecture or function outline.

## Implemented Scope

- Added SQLModel tables for users, conversations, conversation messages, episodes, candidate memory items, profile fields, recent statuses, key proposals, training plan progress, training log imports, training sessions, and memory traces.
- Added `rolling_summary_until_message_id` to conversations so the service can track which raw turns have already been compacted into the rolling summary without deleting message history.
- Added a new `app/memory/` package with approved v1 key catalogs, typed schemas, LLM-backed graph-compatible nodes with deterministic fallbacks, and a `MemoryService` service boundary.
- Added FastAPI memory routes under `/api/memory`.
- Added manual profile editing with approved key validation and explicit extension-mode key proposals.
- Added manual recent-status editing with v1 default `valid_until` estimates.
- Added working memory storage through append-message and conversation-context APIs.
- Added token-based working-memory windowing: when unsummarized working-memory text exceeds the working-memory token budget, the oldest unsummarized messages are summarized until the remaining live raw window is about 30% of that budget.
- Added pre-summary memory extraction so user messages about to leave the live working-memory window create reviewable candidate memory items before the rolling summary cursor advances.
- Added LLM-backed rolling-summary compaction for oversized working-memory windows, with deterministic fallback.
- Added rolling-summary self-compression: when the persisted rolling summary exceeds its own section budget, an LLM rewrites it toward about 50% of that budget when available; deterministic fallback drops the oldest summary lines.
- Added idempotent conversation close behavior that writes one episode per conversation.
- Added lazy inactivity closure for open conversations with no update for more than one hour. The close is triggered when conversations are next listed, inspected, or reused because there is no background scheduler in this block.
- Added closed-conversation reuse behavior: if a user continues a closed conversation ID, the conversation is reopened before new turns are appended.
- Added episode refresh behavior for reopened conversations: closing again updates the existing episode summary/message coverage and extracts candidates only from newly unprocessed messages instead of creating a duplicate episode.
- Added LLM-backed episode summarization and completed-plan episode writing, with deterministic fallbacks.
- Added LLM structured candidate extraction for profile and recent-status memory items, with deterministic fallback for common v1 facts such as goals, lift PRs, sleep, pain, injury, illness, soreness, stress, preferences, dislikes, allergies, height, bodyweight, and training-day availability.
- Added review APIs for approving, rejecting, and expiring candidate memory items.
- Added key-proposal review APIs and LLM duplicate/equivalence review for proposed custom keys, with deterministic nearest-key fallback.
- Added recent-status expiry checks, user-facing expiry reminder payloads, explicit expire-without-delete behavior, audit listing, and editable `valid_until` renewal.
- Added active training-plan progress storage with append-only JSON event history.
- Added completed-plan episode writing when a tracked plan is marked completed.
- Added Xunji training-log import with idempotence by external ID or checksum and simple normalized training sessions.
- Added memory-context retrieval with profile, recent status, episodic memory, working memory, conditional active-plan context, and imported training-log summary sections.
- Profile and recent-status context now keeps only the latest value per key so downstream LLM nodes do not see conflicting duplicate key/value entries.
- Added LLM-backed active-plan context routing with deterministic keyword fallback and global memory-context token-budget warning behavior through settings.
- Added `GET /api/memory/context` so the UI and callers can request full memory context by `user_id` with optional `conversation_id`.
- Added `GET /api/memory/users` so the frontend can show existing local user IDs.
- Added `GET /api/memory/conversations` so the frontend can show existing local conversations.
- Added `GET /api/memory/conversations/{conversation_id}` so the frontend can inspect one conversation with all saved messages.
- Added caller-supplied conversation IDs for memory conversation creation.
- Added explicit delete APIs for user-owned memory rows, conversations, profile fields, and recent-status rows.
- Added profile-key, recent-status-key, current-recent-status, recent-status audit, recent-status expiry reminder, and recent-status expire/renew endpoints for frontend inspection and review.
- Added MCP tools for memory context, episodes, memory items, key proposals, recent status, expiring status, active plan progress, and training-log summary.
- Added a basic Streamlit Memory tab and memory-enabled Q&A controls in the Chat tab. The Memory tab keeps full context retrieval separate from episode search, can inspect saved conversation content, and includes filtered review queues for memory items and key proposals.
- Streamlit ID input fields accept IDs pasted either raw or with surrounding JSON/string quotes, such as `"pf_123"`.
- Added focused memory tests and a Q&A integration test.

## QA Block Changes

The Q&A block was intentionally changed only at the optional memory boundary:

- `AskQuestionRequest` now accepts optional `user_id`, `conversation_id`, `include_memory`, and `memory_debug`.
- `QAService.ask()` accepts the same optional fields.
- When `include_memory=true`, `QAService` asks `MemoryService` for memory context and passes it to answer generation.
- Q&A retrieval from approved evidence remains separate from user memory retrieval.
- The answer prompt labels memory as "personalization only" and says not to treat memory as evidence for scientific claims.
- If `user_id` is supplied, `QAService` ensures the memory conversation exists, auto-generating a `conv_...` ID when `conversation_id` is blank or reopening a closed supplied conversation ID, then best-effort logs the user question and assistant answer through `MemoryService.append_message()`.
- `AskQuestionResponse` now returns the resolved `conversation_id` so the UI can show generated conversation IDs and reuse them for follow-up turns.
- `AnswerTrace` now stores optional `user_id`, `conversation_id`, `memory_context_json`, and `memory_debug_ids_json`.
- Q&A debug output includes memory context and memory IDs when `debug` or `memory_debug` is enabled.
- `README_QA.md` was updated only where the FastAPI route-count smoke test changed because memory routes are now mounted in the same app.

## Assumptions And Gaps

- SQL tables remain the source of truth. Chroma is attempted only as an optional episode-search index when embeddings and Chroma are configured; SQLite keyword search remains the fallback.
- Memory LLM nodes use the same configured chat provider/model path as the Q&A LLM nodes through `OpenAIProvider` and `settings.active_chat_model`. If the chat provider is unavailable, deterministic fallbacks keep local tests and debugging usable without API keys.
- LLM-extracted memory candidates remain reviewable before they affect approved profile or recent-status context. Unknown useful keys create key proposals instead of silently expanding the schema.
- Recent-status expiry reminders are exposed through API and MCP. No background scheduler has been added yet; expired rows are hidden lazily when recent status is read and remain in SQL storage for audit.
- Conflict handling for manual profile edits updates the current approved field directly because the edit is already user-provided. LLM-extracted candidates still require review before they affect profile or status context.
- Plan progress is tracked, but this block does not generate training plans.
- Training-log import is intentionally provider-tolerant because the exact Xunji payload schema was not provided. The raw payload is stored for audit, and common session fields are normalized when present.
- Working-memory windowing uses the configured working-memory section token budget. The rolling-summary cursor prevents already summarized turns from re-entering the live working-memory window, but all raw messages remain stored for audit and conversation inspection.
- Rolling-summary self-compression is intentionally persistent because `Conversation.rolling_summary` is already a lossy working-memory artifact. Raw conversation messages remain stored for audit and future episode creation.
- Pre-summary extraction creates candidates without requiring a closed episode. When the conversation is later closed, those existing candidates are linked to the episode instead of being re-created.
- Memory context retrieval no longer trims or summarizes sections at the final prompt stage. If a section or the whole assembled context exceeds its configured budget, `MemoryService` returns user-visible `budget_warnings` and leaves the context text unchanged.
- Conversation inactivity closure is lazy, not scheduled. It runs when the memory service touches conversations, including listing, detail/context access, and reuse before appending new messages.
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
25 passed
```

- FastAPI app import smoke test passed:

```powershell
.\.venv\Scripts\python.exe -c "from app.api.main import app; print(app.title); print(len(app.routes))"
```

Expected result:

```text
Personal Coach Evidence-Based Q&A Assistant
56
```

- Full test suite passed:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result:

```text
84 passed
```
