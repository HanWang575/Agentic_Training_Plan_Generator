# Program Generator Implementation Notes

## Current Scope

- The active Program Generator is generation-only. It creates new strength-training programs, explains recorded rationale, saves reviewable drafts, and activates approved drafts through Memory.
- The active outline is `docs/functions/program_generation_stepwise_outline.md`.
- The pre-simplification outline is preserved as `docs/functions/program_generator_old.md`.
- Active-plan changes after missed sessions or feedback are intentionally outside the current Program Generator workflow.

## Implemented Design

- A conversation-scoped, checkpointed `ProgramContext` stores the current turn, compact Memory context, latest four weekly summary notes, current-step evidence, clarification answers, SOP progress, `new_program_working_state`, draft artifacts, and safety/budget warnings.
- The graph routes only `new_program`, `clarification_answer`, and `explain_plan_rationale`.
- New programs are written step by step. Deterministic code creates an empty structured `raw_plan`; `SOPStepExecutor` writes each SOP decision; `PlanUpdater` updates the raw plan from that decision.
- `load_calculation` and `volume_calculation` are deterministic. Numeric prescriptions are materialized through calculator traces before a draft can be saved.
- Weighted exercises require a usable Memory 1RM/current user 1RM/recent weighted set reference before `load_kg` is filled. Bodyweight exercises may keep `load_kg=null`.
- `raw_plan` stores the high-level editable plan. Deterministic expansion is stored as `calculation_artifacts`, and finalization reconstructs the saved `ProgramVersion` from `sop_progress.decisions`, `raw_plan`, and `calculation_artifacts`.
- `retrieved_evidence` stores full chunks only for the current SOP step. Historical evidence refs remain compact in `sop_progress`; full audit remains in trace/tool-call/citation tables.
- The stored ProgramContext copy of Memory context omits Memory debug metadata and rendered prompt text. The Memory block response itself is unchanged.

## Interfaces

- FastAPI, MCP, and Streamlit all call the same `ProgramService`.
- Approval sends the complete immutable version snapshot to Memory. Memory owns active-plan status and progress and deactivates older active plans.
- Conversation closure follows the Memory block behavior. If an Episode references a ProgramContext, it stores the ProgramContext ID; the complete ProgramContext remains in Program Generator storage.

## Verification

- Focused Program Generator/provider tests: `.\.venv\Scripts\python.exe -m pytest tests\test_program_service.py tests\test_embedding_provider.py`
