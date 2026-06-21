# Program Generator Block

This repository implements the Program Generator block of the Personal Coach AI Agent alongside the Evidence-Based Q&A Assistant and User Profile And Memory blocks.

The Program Generator creates new training programs, revises active programs, explains recorded design rationale, calculates every visible prescription deterministically, and sends approved versions to Memory for activation and progress tracking. Work is maintained in a checkpointed `ProgramContext` for each open Program Generator conversation.

## Folder Guide

- `app/program/`
  - Program Generator schemas, LangGraph decisions, prompts, deterministic calculators, and orchestration service.
  - Important files:
    - `service.py`: conversation lifecycle, SOP execution, persistence, review, activation, summaries, and traces.
    - `graph.py`: LangGraph construction and fallback runner.
    - `nodes.py`: task routing, SOP-step readiness, next-action routing, and patch validation.
    - `calculators.py`: safe deterministic calculation functions shared by plans, FastAPI, and MCP.
    - `schemas.py`: request, response, graph-state, patch, plan, and trace contracts.
    - `prompts.py`: structured new-program and adaptation prompt.
- `app/api/routes_program.py`
  - FastAPI routes for conversation turns, contexts, versions, review, traces, weekly summaries, calculators, and optional Xunji write-back.
- `app/memory/`
  - Supplies profile, recent status, full active-plan progress, training-log context, conversation storage, Episode creation, and approved-plan activation.
- `app/knowledge/` and `app/qa/`
  - Supply approved evidence through the same services exposed by the Q&A MCP tools.
- `app/storage/models.py`
  - Contains ProgramContext, program artifact, adaptation, summary, trace, citation, calculation, and safety tables.
- `app/mcp_server/server.py`
  - Exposes Program Generator, calculator, Q&A, and Memory tools through FastMCP.
- `app/ui/streamlit_app.py`
  - Contains the dedicated `Program Generator` tab.
- `scripts/refresh_program_weekly_summaries.py`
  - Creates missing summaries for the latest four completed natural weeks. Schedule it on Sundays in production.
- `scripts/start_local.ps1`
  - Starts the API and Streamlit UI in hidden local processes.
- `tests/test_program_service.py`
  - Focused tests for routing, SOP order, clarification, calculations, review, activation, adaptation, safety, summaries, context inheritance, and closure references.
- `docs/functions/program_generator.md`
  - Authoritative Program Generator design outline.
- `docs/implementation_notes/program_generator_notes.md`
  - Implementation decisions, assumptions, instruction gaps, and verification evidence.

The outline files are planning documents and are not modified by the implementation.

## What Is Included

- A dedicated Streamlit `Program Generator` tab.
- A conversation-scoped, revision-checked `ProgramContext`.
- New-program, revision/adaptation, clarification, and rationale routes.
- Ordered new-program and adaptation SOP execution.
- A bounded reasoning/action loop with at most five RAG queries and five user questions per SOP step.
- Approved-evidence retrieval and citations.
- Full Memory context, including the active plan, session IDs, progress, profile, and recent status.
- Stable summaries for the latest four completed Monday-Sunday training weeks.
- Deterministic tools for expressions, repetition maximum, percentage load, load rounding, volume, duration, rest, pace/distance/time, weekly frequency, and progression.
- Immutable draft program versions with dated sessions, exercises, sets, rationale, warnings, provenance, and calculation traces.
- Approve/reject/request-change review behavior.
- Automatic Memory activation after approval, including deactivation of older active plans.
- Progress preservation for unchanged week/day slots when an adaptation replaces an active version.
- Program, tool-call, calculation, citation, and safety traces.
- FastAPI and MCP interfaces that call the same service layer.

## How The Workflow Fits Together

1. The user sends a message with a Program Generator conversation ID.
2. Memory stores the user message.
3. The Program Generator creates or resumes the conversation's `ProgramContext`.
4. The context is hydrated with Memory data and the latest four stored weekly summaries.
5. `TaskAndSOPRouter` chooses new program, adaptation, clarification resume, or rationale explanation.
6. For each SOP step, the graph decides whether to retrieve evidence, ask the user, write the step output, run calculations, or finalize.
7. Every accepted context update is validated, checkpointed, and traced.
8. The finalizer stores an immutable draft version and returns it for review.
9. Approval sends the complete version to Memory, which deactivates older plans and activates the approved version atomically.
10. Memory remains the canonical owner of active-plan status and progress.

A request for changes is sent as a normal conversation turn. It creates a new draft version and never mutates the Memory-owned active plan before approval.

## Local Setup

Run commands from:

```powershell
cd C:\Users\Han\Documents\Codex\2026-06-08\i-want-to-create-an-ai\personal-coach-ai-agent
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the project and create local configuration:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Program generation requires a working chat-model configuration. The app and calculator endpoints still start without one, but generation returns a clear configuration error.

Example chat configuration:

```text
LLM_MODEL_TYPE=aihubmix
LLM_MODEL_ID=gpt-5.4-mini
LLM_API_KEY=your_aihubmix_api_key
LLM_BASE_URL=https://aihubmix.com/v1
```

Approved evidence retrieval also benefits from a configured embedding provider:

```text
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v3
EMBED_API_KEY=your_dashscope_api_key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## Program Configuration

The Program Generator defaults are listed in `.env.example`:

```text
PROGRAM_MAX_RAG_QUERIES=5
PROGRAM_MAX_USER_QUESTIONS=5
PROGRAM_MAX_SOP_STEPS_PER_RUN=12
PROGRAM_MAX_CONTEXT_TOKENS=66000
PROGRAM_CHECKPOINT_AFTER_EACH_CONTEXT_UPDATE=true
PROGRAM_EVIDENCE_CONTEXT_TOP_K=12
PROGRAM_DEFAULT_UNITS=metric

PROGRAM_BARBELL_LOAD_ROUNDING_KG=0.5
PROGRAM_DUMBBELL_LOAD_ROUNDING_KG=2.0
PROGRAM_MACHINE_LOAD_ROUNDING_KG=2.5
PROGRAM_BODYWEIGHT_LOAD_ROUNDING_KG=0.5
PROGRAM_REST_ROUNDING_SECONDS=5
PROGRAM_DURATION_ROUNDING_SECONDS=60
PROGRAM_PACE_ROUNDING_SECONDS_PER_KM=1

PROGRAM_REQUIRE_APPROVAL_BEFORE_ACTIVATION=true
PROGRAM_REQUIRE_APPROVAL_BEFORE_XUNJI_WRITE=true
PROGRAM_XUNJI_WRITE_ENABLED=false
```

Keep approval requirements enabled unless you intentionally change the product's review policy.

## Step-By-Step Tutorial

### 1. Run Syntax And Tests

```powershell
.\.venv\Scripts\python.exe -m compileall -q app tests
.\.venv\Scripts\python.exe -m pytest -q
```

Current expected result:

```text
100 passed
```

The tests do not require a live LLM because Program Generator tests use deterministic fake providers.

### 2. Verify The API And MCP Apps

```powershell
@'
from app.api.main import create_app
from app.mcp_server.server import create_mcp_server

app = create_app()
print(app.title)
print(len(app.routes))
print(len(create_mcp_server()._tool_manager.list_tools()))
'@ | .\.venv\Scripts\python.exe -
```

Current expected result:

```text
Personal Coach AI Agent
68
29
```

### 3. Start The API And UI

The helper starts both services if their ports are free:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_local.ps1
```

Open:

- API health: `http://127.0.0.1:8000/health`
- Swagger: `http://127.0.0.1:8000/docs`
- Streamlit: `http://127.0.0.1:8501`

Logs are written under `tmp/`.

You can also start the services manually:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run app\ui\streamlit_app.py --server.port 8501
```

### 4. Add Useful Profile Data

The Program Generator can ask for missing information, but a populated Memory profile makes generation faster and more personal.

```powershell
$profile = @{
  fields = @{
    primary_training_goal = "strength"
    available_training_days = 3
    typical_session_duration_minutes = 60
    experience_level = "intermediate"
    available_equipment = "full gym, barbell"
    medical_constraints_user_reported = "no known limitations"
    squat_1rm_kg = 140
    bench_press_1rm_kg = 100
    deadlift_1rm_kg = 180
  }
  source = "manual"
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/profile/usr_demo" `
  -Method Patch `
  -ContentType "application/json" `
  -Body $profile
```

Use only measurements and health information the user has actually supplied. The generator does not provide medical clearance.

### 5. Request A New Program

```powershell
$turn = @{
  user_id = "usr_demo"
  message = "Create a four-week strength plan. I can train 3 days per week for 60 minutes, I am intermediate, I have a full gym, and I have no current injuries or medical limitations."
} | ConvertTo-Json

$draft = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/conversations/conv_demo_program/turns" `
  -Method Post `
  -ContentType "application/json" `
  -Body $turn

$draft
```

Possible results:

- `waiting_for_user`: answer the clarification question with another turn using the same conversation ID.
- `waiting_for_review`: the response includes `draft_program_id` and `draft_version_id`.
- `waiting_for_safety_clearance`: normal planning stopped because a red flag requires professional evaluation or clearance.
- HTTP `503`: no working chat model is configured.

### 6. Answer A Clarification

Use the same conversation ID:

```powershell
$answer = @{
  user_id = "usr_demo"
  message = "My main goal is strength. I can train Monday, Wednesday, and Friday. Sessions can be 60 minutes. I am intermediate, have a full gym, and have no current pain or medical limitations."
} | ConvertTo-Json

$draft = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/conversations/conv_demo_program/turns" `
  -Method Post `
  -ContentType "application/json" `
  -Body $answer
```

Clarification answers resume the same ProgramContext and pending SOP step.

### 7. Inspect The Context, Version, And Trace

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/contexts/$($draft.context_id)"

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/$($draft.draft_program_id)/versions/$($draft.draft_version_id)"

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/traces/$($draft.trace_id)"
```

The version contains dated sessions, ordered exercises, set prescriptions, assumptions, warnings, source references, and calculation trace IDs. The context contains SOP progress, evidence, working state, draft references, and safety/budget warnings.

### 8. Request A Change Or Adaptation

Send requested changes through the conversation endpoint rather than the review endpoint:

```powershell
$change = @{
  user_id = "usr_demo"
  message = "Revise the draft so Wednesday is a shorter 40-minute session while preserving the weekly strength goal."
} | ConvertTo-Json

$revision = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/conversations/conv_demo_program/turns" `
  -Method Post `
  -ContentType "application/json" `
  -Body $change
```

For a missed session, state why it was missed. The adaptation SOP distinguishes schedule conflicts from recovery, illness, pain, or fatigue and adjusts frequency, volume, intensity, exercise selection/order, and rest in that priority order.

### 9. Approve And Activate A Draft

```powershell
$review = @{
  user_id = "usr_demo"
  decision = "approve"
  reason = "Reviewed and accepted"
} | ConvertTo-Json

$approved = Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/$($draft.draft_program_id)/versions/$($draft.draft_version_id)/review" `
  -Method Post `
  -ContentType "application/json" `
  -Body $review

$approved
```

Approval automatically activates the complete plan through Memory. Older active plans for the user become `superseded`. If activation fails, the response reports `activation_failed` and the approved version can be retried.

To reject instead:

```powershell
$review = @{
  user_id = "usr_demo"
  decision = "reject"
  reason = "Needs another revision"
} | ConvertTo-Json
```

### 10. Inspect The Memory-Owned Active Plan

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/memory/plans/active?user_id=usr_demo"
```

Memory owns active-plan status and session progress. The Program Generator owns immutable generated versions and adaptation records.

### 11. Ask For The Plan Rationale

```powershell
$why = @{
  user_id = "usr_demo"
  message = "Explain why this plan was designed this way."
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/conversations/conv_demo_program/turns" `
  -Method Post `
  -ContentType "application/json" `
  -Body $why
```

Rationale responses read the recorded SOP decisions and linked support. They do not create a new SOP or program version.

### 12. Refresh And Read Weekly Summaries

Context preparation reads summaries but never creates them inline. Run the scheduled operation separately:

```powershell
.\.venv\Scripts\python.exe scripts\refresh_program_weekly_summaries.py
```

Or refresh one user through the API:

```powershell
$summaryRequest = @{ user_id = "usr_demo" } | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/training/summary-notes/refresh" `
  -Method Post `
  -ContentType "application/json" `
  -Body $summaryRequest

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/training/summary-notes?user_id=usr_demo&limit=4"
```

Completed weekly notes are stable and use the user's timezone.

### 13. Try A Calculator Without An LLM Key

```powershell
$calculation = @{
  inputs = @{
    one_rm = 100
    percentage = 75
    unit = "kg"
  }
  source_references = @("manual_demo_1rm")
} | ConvertTo-Json -Depth 4

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/programs/calculators/calculate_percentage_load" `
  -Method Post `
  -ContentType "application/json" `
  -Body $calculation
```

Expected value: `75 kg`. The response also contains the method, inputs, rounding policy, source references, and tool-call ID.

### 14. Use The Streamlit Tab

Open `http://127.0.0.1:8501` and select `Program Generator`.

The tab provides:

- New-plan, revision, clarification, and rationale conversation turns.
- The Memory-owned active plan and progress.
- Current SOP progress and waiting state.
- Draft/version inspection.
- Approve, reject, and request-change controls.
- Evidence, calculation, safety, warning, and trace details.
- Weekly-summary refresh.
- Separately confirmed Xunji write-back when enabled.

The UI calls backend APIs and does not contain planning or calculation policy.

### 15. Run The MCP Server

```powershell
.\.venv\Scripts\python.exe -m app.mcp_server.server
```

Program tools:

- `send_program_conversation_turn`
- `get_program_context`
- `get_program_version`
- `review_training_program`
- `get_program_trace`
- `write_training_program_to_xunji`

Calculator tools:

- `calculate_expression`
- `estimate_repetition_max`
- `calculate_percentage_load`
- `round_training_load`
- `calculate_volume_load`
- `calculate_session_duration`
- `calculate_rest_interval`
- `calculate_distance_pace_time`
- `calculate_weekly_frequency`
- `calculate_progression_increment`

The Program Generator also consumes `search_knowledge`, `ask_qa_assistant`, and `get_user_memory_context` from the other blocks.

## Safety And Review Rules

- Red-flag symptoms stop normal planning.
- The generator asks for raw inputs and does not ask the user to calculate training values.
- Missing inputs after the question limit become conservative, disclosed assumptions.
- Missing approved evidence after the retrieval limit produces an explicit warning.
- Every visible numeric prescription is calculated or validated deterministically and linked to calculation traces.
- Drafts do not become active until approval.
- Active-plan progress is never owned or edited directly by the Program Generator.
- Conversation messages are stored through Memory, and a closing Episode references the ProgramContext rather than copying it.

## Known V1 Limitations

- Xunji data import is available through the Memory block, but the repository's supplied Xunji integration is read-only. Even when Program write-back is enabled, the endpoint returns `unavailable` until a supported write endpoint and payload contract exist.
- The graph calls shared Q&A and Memory services in-process. MCP exposes the same service contracts for external callers.
- The deterministic evidence-sufficiency rule treats one approved retrieved chunk as sufficient for the current SOP step.
- The weekly-summary script must be scheduled by the deployment environment.
- Without a matching user-supplied 1RM, resistance prescriptions use traced percentage/RPE guidance without inventing a kilogram load.

See `docs/implementation_notes/program_generator_notes.md` for the complete assumptions and instruction-gap record.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
