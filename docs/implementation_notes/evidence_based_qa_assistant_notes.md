# Evidence-Based Q&A Assistant Implementation Notes

This file records implementation assumptions and instruction gaps without changing the original architecture or function outline.

## Implemented Scope

- Added the suggested `app/` and `tests/` structure.
- Added typed settings loaded from environment variables and `.env`.
- Added SQLModel tables for `source_documents`, `knowledge_chunks`, `answer_traces`, `citations`, and `safety_flags`.
- Added SQLite FTS5 keyword indexing for approved chunks.
- Added Chroma vector indexing and retrieval hooks through an OpenAI provider adapter.
- Added parsers for `.pdf`, `.epub`, `.md`, `.txt`, and `.json`.
- Added upload, pending review, approval, rejection, source listing, search, ask, and trace APIs.
- Added a node-based Q&A workflow with input validation, safety classification, query planning, retrieval, RRF merging, context building, answer generation, citation checking, and trace logging.
- Added MCP tools that call the same services as the backend.
- Added a thin Streamlit UI that calls backend APIs.

## Assumptions And Gaps

- The current `RAG_corpus/` folder was not moved, modified, or auto-indexed because the detailed function outline says it is available source material for future upload/indexing work and does not require indexing yet.
- The outline names `gpt-5.4-mini` and `gpt-5.5`; those names are kept in configuration as requested. If the account does not support them, change `.env` without editing graph nodes.
- No API key is required for the app to start. Semantic indexing, semantic retrieval, and LLM answer generation report a clear `OPENAI_API_KEY` configuration error when used without a key.
- Approval returns `index_error` when semantic indexing cannot run because the API key is missing. This keeps trusted approved knowledge aligned with the rule that approved sources should be indexed into both SQLite FTS5 and Chroma.
- The Q&A service runs the node workflow through LangGraph when LangGraph is installed, and falls back to the deterministic node runner only when the optional package is unavailable.
- Neighbor-chunk expansion is implemented as a deterministic graph node. It adds up to one previous and one next approved chunk from the same source before the context builder applies the maximum context token cap.
- Citation checking verifies cited source/chunk IDs against retrieved context. It does not yet perform claim-level semantic entailment.
- Runtime verification was completed after Python 3.11 became available: `compileall` passed for `app` and `tests`, the focused pytest suite passed, and the FastAPI app imported successfully with 12 routes.
- The full optional dependency install is slow in this environment because large packages such as Chroma download at low bandwidth. Focused tests were run with the smaller dependency subset they require.
- DashScope / Aliyun embeddings are supported through OpenAI-compatible `EMBED_*` settings. A live smoke test with `EMBED_MODEL_TYPE=dashscope`, `EMBED_MODEL_NAME=text-embedding-v3`, and `EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1` successfully uploaded, approved, indexed, and searched a one-chunk test source.
- Aihubmix chat generation is supported through OpenAI-compatible `LLM_*` settings. When `LLM_API_KEY` or `LLM_BASE_URL` is present, answer generation uses `LLM_MODEL_ID` and `LLM_BASE_URL` through chat completions while embeddings continue to use `EMBED_*`.
