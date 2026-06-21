# Program Generator Implementation Notes

## Implemented Scope

- Added a conversation-scoped, checkpointed `ProgramContext` and the program artifact, adaptation, trace, tool-call, calculation, citation, safety-flag, and weekly-summary tables from the outline.
- Added a typed LangGraph decision flow for task/SOP routing, SOP-step readiness, and the next reasoning action. The bounded service loop advances one SOP step at a time through evidence retrieval, clarification, working-state updates, deterministic calculation, and finalization.
- Added ordered new-program and adaptation SOP records. Each completed SOP step updates the working-state section and `sop_progress`, then checkpoints and traces the new context revision.
- Added persisted per-SOP-step RAG and user-question counters with the configured limit of five. Reaching a clarification limit produces conservative, disclosed defaults; reaching the evidence limit produces an insufficient-evidence warning.
- Added deterministic calculator services and matching FastAPI/MCP tools. Session duration, frequency, sets, reps, intensity, RPE, rest, progression, training load when a matching 1RM exists, per-exercise volume, and weekly volume carry calculation traces.
- Added immutable draft/revision versions with dated sessions, ordered exercises, complete set prescriptions, rationale, assumptions, warnings, evidence/memory references, and calculation references. Resistance and aerobic prescriptions use the same deterministic calculator service.
- Added review behavior. Approval automatically sends the complete immutable snapshot to Memory, atomically deactivates older active plans, and activates the approved version. Activation failure is explicit and retryable; revision activation preserves progress for unchanged week/day slots.
- Added a dedicated Program Generator Streamlit tab with conversation, active-plan progress, SOP/evidence/calculation/trace inspection, approve/reject/request-change controls, and separately confirmed Xunji write-back. The FastAPI and MCP interfaces call the same services.
- Added natural-week summary persistence, retrieval, an explicit refresh API, and `scripts/refresh_program_weekly_summaries.py` for a Sunday scheduler. Notes use the user's timezone and cover adherence, emphasis, workload/intensity, recovery/safety, performance references, plan changes, and source IDs.
- A new conversation that revises or explains the active plan creates a child ProgramContext linked to the active version's prior audit context. Mutable planning starts fresh for a revision while prior evidence, calculations, and compact SOP history remain available.
- Extended Memory only where the Program Generator contract required it: atomic activation, active `version_id`, complete active-plan snapshot, and an optional Episode-to-ProgramContext reference.
- Added `README_PROGRAM_GENERATOR.md` with the block architecture, setup, configuration, API/MCP/UI tutorial, safety rules, tests, and known v1 limitations.

## Assumptions And Instruction Gaps

- Sections 1-4 of `program_generator.md` are authoritative. Section 4.2 currently says a new-program request refreshes the existing conversation context, while the test list still says it creates a new context. The implementation keeps one context per open Program Generator conversation and clears only program-design sections for a new plan.
- Earlier outline drafts named an `EvidenceSufficiencyJudge`; the authoritative current contracts use `SOPStepReadinessAssessor`. V1 treats at least one approved retrieved chunk as sufficient for the current step; otherwise the graph may query again up to the per-step limit.
- The outline says cross-block evidence and memory calls use MCP. Because all blocks run in one Python application and MCP is a thin wrapper over the same services, internal graph calls use those shared services directly; external callers receive the matching MCP tools.
- Memory's implemented context API does not yet expose the outline's `context_profile` argument. The Program Generator requests `task_type=program_generator`, which the Memory plan router treats as full active-plan context.
- The LLM proposes exercise selection and programming decisions, but deterministic code bounds and recalculates every numeric prescription before persistence. Missing 1RM data produces percentage/RPE prescriptions without inventing a kilogram load.
- Weekly notes are stable once created and are never generated during context preparation. The repository supplies a refresh script and endpoint, but operating-system scheduling remains a deployment responsibility.
- Xunji's repository integration and supplied API information are read-only. The Program Generator therefore enforces separate confirmation and configuration, but returns `unavailable` without sending data when write-back is enabled. No write endpoint or payload schema was invented.
- The outline requires ProgramContext to be referenced, not copied, by a closing Episode. Memory now accepts `program_context_id` on closure; the complete context remains in Program Generator storage.
- Context token limits use the repository's existing deterministic character-based estimate. When exceeded, the workflow pauses and requires an explicit `continue` or `stop`; protected content is not silently trimmed.

## Verification

- `python -m compileall -q app tests`
- `pytest tests/test_program_service.py -q`: 11 focused contract tests pass.
- `pytest -q`: 100 repository tests pass with the final audit assertions included.
- FastAPI OpenAPI construction exposes all eight Program Generator/training-summary paths, and FastMCP construction registers 29 shared tools.
- The local API health endpoint returned `{"status":"ok"}` and the dedicated Streamlit tab rendered the conversation, review/change controls, SOP context, and trace surfaces.
