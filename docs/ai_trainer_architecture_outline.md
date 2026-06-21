# AI Trainer Project Outline Plan

## Summary

The AI trainer will be organized as a modular coaching system made of independent function blocks. The Evidence-Based Q&A Assistant provides the shared evidence layer through user-uploaded materials, trusted references, and RAG. The User Profile And Memory block stores personalized user context, the approved active plan, plan progress, and training-log history with recency so other blocks can retrieve them when needed. The Program Review/Generator block creates, reviews, and adapts training plans, then sends approved plans to the Memory block for activation and progress tracking. External Integrations remain optional adapters for future tools such as wearables, calendars, and document sources.

Detailed schemas, APIs, prompts, LangGraph workflows, tests, and implementation notes for each block should live in separate dedicated files later.

## Implementation Principle

Build the system step by step in the roadmap order: Evidence-Based Q&A Assistant, User Profile And Memory, Training Log, Check-In, And Program Review/Generator, and External Integrations. Each block should work as an independent service before later blocks depend on it.

## Shared Design Rules

- Each function block owns its own data.
- Other blocks should access data through stable service interfaces instead of reading another block's internal storage directly.
- Human-editable data is preferred over hidden memory whenever the information may affect training, nutrition, or safety.
- LLM outputs that affect trusted knowledge, user memory, or active training plans should be reviewable by the user.
- The system should make it visible which evidence, memory, logs, and generated plans were used to produce important recommendations.

## Debugging Strategy

- Log retrieved sources for every evidence-based answer.
- Log memory items extracted from each conversation, including category and recency.
- Log which function blocks were called for each major response.
- Show the user which evidence and memory were used when generating training plans or personalized advice.

## Function Roadmap

### 1. Evidence-Based Q&A Assistant

- Purpose: provide evidence-based answers using trusted references and user-approved materials.
- Core functions:
  - Allow the user to upload related materials to its knowledge base.
  - Answer questions about strength training, bodybuilding, powerlifting, nutrition, HYROX-style hybrid training, and other sports the user is interested in.
- Main stack: LangGraph, hybrid RAG, keyword retrieval, semantic retrieval, FastAPI, vector database, document parser.
- Data and pipelines: trusted references, uploaded materials, document chunks, metadata, embeddings, citations, question logs; document upload, pending review, approval, query understanding, retrieval, context assembly, answer generation, citation grounding, safety check.
- Decoupling boundary: expose as a standalone Q&A/RAG service so other blocks can request evidence without depending on retrieval internals.
- Safety and quality boundary: uploaded materials should stay in pending review until approved by the user, and only approved materials should become trusted knowledge for evidence-based answers.

### 2. User Profile And Memory

- Purpose: maintain a structured, recency-aware personalization layer for the whole AI trainer.
- Core functions:
  - Store meaningful user-related information spotted by an LLM agent during conversation, such as recent sleep issues, injuries, disliked foods, current bench press PR, and any other information useful for planning exercise, relaxation, or food.
  - Store time recency for remembered information.
  - Retrieve stored information whenever needed by other blocks, such as Training Log, Check-In, And Program Review/Generator.
- Main stack: LangGraph memory/checkpointing, SQLite or Postgres, profile service, structured data models, memory extraction agent.
- Data and pipelines: memory categories such as profile, preference, constraint, injury or health note, performance metric, recent status, and goal; goals, training history, equipment, schedule, preferences, limitations, injuries, disliked foods, recent issues, performance PRs, timestamps; memory extraction, classification, storage, review, retrieval, and recency-aware context selection.
- Decoupling boundary: keep user memory separate from the evidence knowledge base and expose it through a stable memory retrieval service.
- Safety and quality boundary: user memory should be visible and editable, especially when it may affect training, nutrition, recovery, or safety recommendations.

### 3.Program Generator

- Purpose: create, review, and adapt training plans using user context, active-plan progress, training summaries, deterministic calculations, and retrieved evidence. Approved plans are handed to the User Profile And Memory block for activation and progress tracking.
- Core functions:
  - Keep summary training notes for every week.
  - Generate a professional training plan based on information provided by the user, the User Profile And Memory block, training summaries, direct readiness/recovery feedback, and the professional knowledge retrieved from the RAG system of the Evidence-Based Q&A Assistant block.
  - Explain the rationale behind the generated training plan.
  - Modify the training plan according to user's feedback. 
  - Send the complete approved plan to the User Profile And Memory block for activation and progress tracking, and write it to Xunji through API only after separate user approval.
  - When current information is not sufficient to generate a professional training plan, ask the user for extra information such as goal, current training level, and number of days available in the coming week.
  - If the user is unable to achieve the plan because of sickness, meetings, poor sleep, insufficient training level, or other feedback, update and modify the training plan professionally using the user's feedback plus information from the relevant memory, log, and evidence sources.
- Main stack: FastAPI, database tables for generated plan artifacts and traces, simple editing UI, scheduled summary jobs, LangGraph planning workflow, structured output validation, program template library, RAG integration, memory retrieval, and a safety/review layer.
- Data and pipelines: user profile, active-plan progress, training-log summaries, goals, constraints, availability, generated plan versions, evidence references, and feedback; information sufficiency checks, clarification, program drafting, rationale generation, volume/intensity review, progression planning, adaptation, approval, and Memory-block handoff.
- Decoupling boundary: consume profile, active plan, progress, training summaries, and Q&A/RAG evidence through stable interfaces; keep program generation, review, adaptation, and their audit records behind this block's own service interface.
- Safety and quality boundary: generated plans should start as drafts and may be sent to the User Profile And Memory block for activation only after the user reviews and approves them.

### 4. External Integrations

- Purpose: connect optional external tools such as wearables, calendars, documents, or future data sources.
- Core functions: `[To be defined by project owner]`
- Main stack: MCP, integration adapters, sync jobs, import validators.
- Data and pipelines: integration-specific imported data, sync metadata, permissions, source provenance; connector tools, data import, normalization, sync monitoring.
- Decoupling boundary: each integration should be optional and replaceable; core coaching functions should continue working without integrations.
