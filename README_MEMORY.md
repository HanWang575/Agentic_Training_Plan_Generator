# User Profile And Memory

This repository now includes the second function block: the User Profile And Memory system.

The block stores local, inspectable personalization data for the personal trainer AI. It keeps working conversation memory, episodic summaries, reviewable candidate memories, approved profile fields, short-lived recent status, active plan progress, and imported training logs. The Q&A block can optionally request memory context without reading memory tables directly.

## Folder Guide

- `app/memory/`
  - Memory key catalogs, typed API schemas, graph-compatible nodes, and the service layer.
- `app/memory/prompts.py`
  - JSON-only prompts for LLM-backed memory routing, summarization, extraction, and key review.
- `app/api/routes_memory.py`
  - FastAPI routes under `/api/memory`.
- `app/storage/models.py`
  - SQLModel tables for memory storage.
- `app/mcp_server/server.py`
  - MCP tools for memory retrieval and debug access.
- `app/ui/streamlit_app.py`
  - Adds a Memory tab and memory-enabled Q&A controls.
- `tests/test_memory_service.py`
  - Focused tests for profile, recent status, conversation close, deletion controls, candidate review, plan progress, and Xunji import.
- `tests/test_qa_memory_integration.py`
  - Verifies that Q&A can receive memory context and log Q&A turns.
- `docs/implementation_notes/user_profile_and_memory_notes.md`
  - Implementation assumptions, instruction gaps, QA changes, and verification notes.

## What Is Included

- Local SQLite source of truth at `data/app.sqlite`.
- User creation and memory-backed conversations.
- Append-message API for working memory.
- Token-based working-memory window: when unsummarized conversation turns exceed the working-memory budget, the oldest turns are folded into the rolling summary until the live raw window is about 30% of that budget.
- Pre-summary memory extraction: before old working-memory turns are folded into the rolling summary, the same user turns are processed into reviewable candidate memory items so useful profile/status facts are not hidden by summarization.
- LLM-backed rolling summary for oversized working-memory windows, with deterministic fallback and LLM-backed self-compression for oversized rolling summaries. If the compression LLM is unavailable or still returns too much text, the deterministic fallback drops oldest summary content.
- Conversation close API that writes one episode idempotently, updates the episode after reopened conversations add new turns, and extracts candidates only from newly unprocessed messages.
- Lazy one-hour inactivity closure for open conversations when they are next listed or reused.
- LLM-backed episode summaries, completed-plan episodes, and candidate memory extraction, with deterministic fallbacks.
- Human review flow for candidate memory items.
- Approved profile fields and active recent status retrieval.
- Recent-status default expiration, expiry reminders, renewal, and audit listing.
- Extendable profile/status key proposals.
- Debug listing for users and conversations.
- Explicit delete routes for user-owned memory, conversations, profile entries, and recent-status entries.
- Active plan progress tracking with append-only events.
- Xunji import storage and simple training-session normalization.
- Memory context retrieval for Q&A and future blocks.
- LLM-backed active-plan context routing with deterministic fallback.
- Memory context token-budget warnings when assembled sections exceed their configured budgets.
- Optional Chroma episode indexing when embeddings are configured.
- MCP tools that call the same `MemoryService` methods.

## Local Setup

Run these commands from:

```powershell
cd C:\Users\Han\Documents\Codex\2026-06-08\i-want-to-create-an-ai\personal-coach-ai-agent
```

Activate the existing virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If the virtual environment does not exist yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

The memory block uses the same configured chat model as the Q&A LLM nodes when `LLM_*` or `OPENAI_API_KEY` settings are available. It can still be tested without LLM or embedding API keys because memory LLM nodes fall back to deterministic behavior. Episode semantic search and episode Chroma indexing only run when embeddings are configured.

## Step-By-Step Tutorial: Test The Memory Block

### 1. Run A Syntax Check

```powershell
.\.venv\Scripts\python.exe -m compileall -q app tests
```

Expected result: no output and exit code `0`.

### 2. Run The Memory Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_memory_service.py tests\test_qa_memory_integration.py -q
```

Expected result:

```text
25 passed
```

### 3. Run The Full Suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result:

```text
84 passed
```

### 4. Verify The API App Imports

```powershell
.\.venv\Scripts\python.exe -c "from app.api.main import app; print(app.title); print(len(app.routes))"
```

Expected behavior: the app imports successfully and includes the Q&A, knowledge, and memory routes.

### 5. Start The Backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 6. Create A User

```powershell
$user = @{
  user_id = "usr_local"
  display_name = "Local Athlete"
  timezone = "Asia/Shanghai"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/users" `
  -Method Post `
  -ContentType "application/json" `
  -Body $user
```

### 7. Add Profile Fields

```powershell
$profile = @{
  fields = @{
    primary_training_goal = "build strength"
    bench_press_1rm_kg = 100
    preferred_training_days_per_week = 4
  }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/profile/usr_local" `
  -Method Patch `
  -ContentType "application/json" `
  -Body $profile
```

Unknown keys are rejected by default. To propose a new key instead:

```powershell
$profile = @{
  fields = @{
    favorite_barbell = "Texas power bar"
  }
  extension_mode = $true
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/profile/usr_local" `
  -Method Patch `
  -ContentType "application/json" `
  -Body $profile
```

Review proposals:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/key-proposals?user_id=usr_local"
```

Approve a key proposal:

```powershell
$proposalReview = @{} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/key-proposals/YOUR_PROPOSAL_ID/approve" `
  -Method Post `
  -ContentType "application/json" `
  -Body $proposalReview
```

### 8. Add Recent Status

```powershell
$status = @{
  statuses = @{
    sleep_duration_hours = 5.5
    joint_pain = "mild right knee pain after squats"
  }
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/recent-status/users/usr_local" `
  -Method Patch `
  -ContentType "application/json" `
  -Body $status
```

Expected behavior:

- `sleep_duration_hours` gets a default `valid_until` of observed time plus 1 day.
- `joint_pain` gets a default `valid_until` of observed time plus 7 days.
- Both values are visible in memory context until they expire.

Check statuses close to expiry:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/recent-status/expiring?user_id=usr_local"
```

Expected behavior:

- Rows within 2 hours of `valid_until` are returned in `reminders`.
- Each reminder includes `status_key`, `value`, `source`, `current_valid_until`, a prompt, and `suggested_valid_until` when renewing that same value is reasonable.
- Daily observations such as `sleep_duration_hours` do not get a renewal suggestion; record a fresh value instead.

Expire a status without deleting its audit row:

```powershell
$expire = @{
  reason = "no longer true"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/recent-status/YOUR_STATUS_ID/expire" `
  -Method Post `
  -ContentType "application/json" `
  -Body $expire
```

Renew a status by setting a new `valid_until`:

```powershell
$renew = @{
  valid_until = "2026-06-17T12:00:00Z"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/recent-status/YOUR_STATUS_ID/valid-until" `
  -Method Patch `
  -ContentType "application/json" `
  -Body $renew
```

Review active and expired status rows:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/recent-status/current?user_id=usr_local&include_expired=true"
```

If you do not respond before `valid_until`, the row is hidden from normal memory context the next time recent status is read, but remains available through the audit listing.

### 9. Create A Conversation And Append Messages

```powershell
$conversation = @{
  user_id = "usr_local"
  title = "Morning check-in"
  topic = "training readiness"
} | ConvertTo-Json

$created = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/conversations" `
  -Method Post `
  -ContentType "application/json" `
  -Body $conversation

$conversationId = $created.conversation_id

$message = @{
  role = "user"
  content = "My goal is build muscle. My bench press 1RM is 100 kg. I slept 5 hours and I have knee pain today."
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/conversations/$conversationId/messages" `
  -Method Post `
  -ContentType "application/json" `
  -Body $message
```

To choose your own conversation ID instead of using a generated `conv_...` ID, add `conversation_id` to the request body:

```powershell
$conversation = @{
  user_id = "usr_local"
  conversation_id = "250pt"
  title = "Q&A chat"
  topic = "qa"
} | ConvertTo-Json
```

Show the full saved conversation by ID:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/conversations/$conversationId"
```

### 10. Close The Conversation

```powershell
$close = @{
  closure_reason = "test_complete"
  extract_candidates = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/conversations/$conversationId/close" `
  -Method Post `
  -ContentType "application/json" `
  -Body $close
```

Expected behavior:

- One episode is created.
- Candidate memory items are created for review.
- Running close again does not create a duplicate episode.

List candidates:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/items?user_id=usr_local&status=candidate"
```

The `status` query parameter can be omitted to show any status, or set to `approved`, `candidate`, or `rejected`.

Approve a candidate:

```powershell
$approve = @{} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/items/YOUR_MEMORY_ID/approve" `
  -Method Post `
  -ContentType "application/json" `
  -Body $approve
```

Reject a candidate:

```powershell
$reject = @{
  review_note = "Not useful right now."
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/items/YOUR_MEMORY_ID/reject" `
  -Method Post `
  -ContentType "application/json" `
  -Body $reject
```

### 11. Retrieve Memory Context

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/context?user_id=usr_local&conversation_id=$conversationId&query=Can%20I%20train%20today%3F&memory_debug=true"
```

Expected behavior:

- `sections.profile` includes approved profile facts.
- `sections.recent_status` includes active, non-expired recent status.
- `sections.episodic_memories` includes matching prior episodes.
- `memory_ids` lists the rows used to build context.
- `trace_id` starts with `mtr_`.

### 12. Track Active Plan Progress

```powershell
$progress = @{
  user_id = "usr_local"
  planned_sessions = @(
    @{
      session_id = "week1_day1"
      title = "Lower body strength"
    }
  )
  session_id = "week1_day1"
  session_status = "pending"
  event_type = "plan_created"
} | ConvertTo-Json -Depth 6

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/plans/plan_local/progress" `
  -Method Patch `
  -ContentType "application/json" `
  -Body $progress
```

Ask for full plan context:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/context?user_id=usr_local&conversation_id=$conversationId&query=Can%20you%20adapt%20today%27s%20workout%3F&memory_debug=true"
```

Expected behavior: `plan_context_level` is `full`.

### 13. Import A Xunji Training Log Payload

```powershell
$import = @{
  user_id = "usr_local"
  external_id = "xunji_batch_1"
  payload = @{
    sessions = @(
      @{
        id = "sess_1"
        date = "2026-06-12"
        sport_type = "strength"
        exercises = @(@{ name = "Squat"; sets = 3 })
        rpe = 7
        notes = "Felt solid."
      }
    )
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/imports/xunji" `
  -Method Post `
  -ContentType "application/json" `
  -Body $import
```

Run the same request again. Expected behavior: `skipped_duplicate` is `true`.

Retrieve a log summary:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/training-log/summary?user_id=usr_local&days=30"
```

### 14. Test Memory Together With Q&A

Use the `conversation_id` from step 9, or omit `conversation_id` to let Q&A create a new `conv_...` memory conversation automatically when `user_id` is provided:

```powershell
$ask = @{
  question = "How should I think about protein after lifting?"
  topic_hint = "nutrition"
  debug = $true
  user_id = "usr_local"
  conversation_id = $conversationId
  include_memory = $true
  memory_debug = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/qa/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body $ask
```

Expected behavior:

- Q&A still retrieves approved evidence through the Q&A knowledge block.
- `debug.memory_context` shows the memory payload and any token-budget warnings.
- `debug.memory_debug_ids` lists memory rows used.
- The response and Q&A trace store `user_id`, `conversation_id`, memory context, and memory IDs.
- The conversation receives a user message and an assistant message with `source_block="qa"`.

If no approved evidence exists, Q&A still returns its normal insufficient-evidence answer. Memory context does not replace evidence.

### 15. Optional Cleanup: List And Delete Local Memory Rows

List existing users:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/users"
```

List conversations for one user:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/conversations?user_id=usr_local"
```

Load profile and recent status rows before deleting specific entries:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/profile/usr_local"

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/recent-status/current?user_id=usr_local"
```

Delete exact rows by ID:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/profile-fields/YOUR_PROFILE_FIELD_ID" `
  -Method Delete

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/recent-status/YOUR_RECENT_STATUS_ID" `
  -Method Delete

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/conversations/$conversationId" `
  -Method Delete
```

Delete a user and all user-owned memory rows:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/users/usr_local" `
  -Method Delete
```

Run user deletion last if you still need `usr_local` for later tutorial steps.

### 16. Try The Streamlit UI

Start the backend first, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py --server.port 8501 --server.maxUploadSize 400
```

Open:

```text
http://localhost:8501
```

Use:

- `Memory`: create a user, show existing user IDs, delete a user and its memory rows, create/list/show/delete memory conversations, edit profile fields, show profile keys, delete profile entries by ID, edit recent status, load current recent status, show recent-status keys, load expiry reminders, expire or renew recent-status entries, load recent-status audit rows, delete recent-status entries by ID, build full memory context, search episodes, filter memory items by status, approve or reject candidate memory items, load key proposals by status, approve key proposals, and inspect review queues.
- `Chat`: provide a memory user ID, optionally leave the memory conversation ID blank to auto-create one, enable memory context when needed, and ask Q&A requests that include memory. If a memory conversation ID is provided but does not exist yet, Q&A creates that conversation before logging the turn. If the ID belongs to a closed conversation, Q&A reopens it and continues logging new turns.

## MCP Tools

The MCP server exposes these memory tools:

- `get_user_memory_context`
- `search_user_episodes`
- `list_user_memory_items`
- `list_memory_key_proposals`
- `get_recent_status`
- `list_expiring_recent_status`
- `get_active_plan_progress`
- `get_training_log_summary`

They call the same `MemoryService` used by the FastAPI routes.
