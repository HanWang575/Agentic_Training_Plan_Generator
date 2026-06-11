# Personal Coach AI Agent

This repository currently implements the first function block: the Evidence-Based Q&A Assistant.

The assistant is designed as a local-first evidence layer for a future personal trainer AI. It can upload source files, keep them in review, approve or reject them, index approved material, retrieve evidence, generate cited answers, and store debug traces.

## Folder Guide

- `app/`
  - Main Python application package.
- `app/api/`
  - FastAPI app and HTTP routes.
  - Important files:
    - `main.py`: creates the FastAPI app.
    - `routes_qa.py`: Q&A endpoints.
    - `routes_knowledge.py`: upload, source review, source listing, and search endpoints.
- `app/core/`
  - Shared configuration, error types, and OpenAI adapter code.
  - `config.py` loads environment settings from `.env`.
  - `llm_provider.py` wraps embeddings and answer generation.
- `app/knowledge/`
  - Document parsing, chunking, indexing, retrieval, and source review logic.
  - This is where uploads become chunks and approved chunks become searchable evidence.
- `app/qa/`
  - Q&A workflow schemas, graph runner, graph nodes, prompts, and service layer.
  - The workflow validates input, applies safety handling, plans search, retrieves evidence, expands neighbor chunks, builds context, generates an answer, checks citations, and stores a trace.
- `app/storage/`
  - SQLite database setup and SQLModel tables.
  - Tables include `source_documents`, `knowledge_chunks`, `answer_traces`, `citations`, and `safety_flags`.
- `app/mcp_server/`
  - MCP tools for future agent blocks.
  - Tools include `ask_qa_assistant`, `search_knowledge`, `list_approved_sources`, and `get_answer_trace`.
- `app/ui/`
  - Streamlit UI.
  - Provides local chat, upload, review, source listing, and trace lookup views.
- `docs/`
  - Project planning documents and implementation notes.
  - `docs/ai_trainer_architecture_outline.md`: overall project architecture.
  - `docs/functions/evidence_based_qa_assistant.md`: detailed requirements for this block.
  - `docs/implementation_notes/evidence_based_qa_assistant_notes.md`: assumptions, gaps, and verification notes from implementation.
- `RAG_corpus/`
  - Existing source material for future ingestion.
  - The implementation does not modify or auto-index this folder.
- `tests/`
  - Focused pytest tests for upload/review behavior, retrieval merging, neighbor expansion, safety handling, insufficient-evidence behavior, citation checking, and trace logging.
- `data/`
  - Local runtime data created when the app runs.
  - Ignored by git. It can contain SQLite data, uploads, and Chroma data.
- `.venv/`
  - Local Python virtual environment.
  - Ignored by git.
- `.env.example`
  - Example environment configuration.
- `pyproject.toml`
  - Python project metadata, dependencies, and pytest settings.
- `.gitignore`
  - Keeps local virtual environments, caches, local data, `.env`, and build metadata out of version control.

## What Is Included

- FastAPI backend at `app/api/main.py`
- Streamlit UI at `app/ui/streamlit_app.py`
- MCP tools at `app/mcp_server/server.py`
- SQLite metadata storage with FTS5 keyword search
- Chroma vector indexing hooks
- LangGraph Q&A workflow with a simple fallback runner
- Upload, bulk `RAG_corpus` import, review, approve, reject, remove, search, ask, and trace lookup workflows
- Focused pytest tests under `tests/`

## Local Setup

Run these commands from:

```powershell
cd C:\Users\Han\Documents\Codex\2026-06-08\i-want-to-create-an-ai\personal-coach-ai-agent
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the full project:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Add API keys to `.env` before running semantic indexing or real answer generation.

Chat-answer calls are configured separately from embedding calls. To use aihubmix for answer generation:

```text
LLM_MODEL_TYPE=aihubmix
LLM_MODEL_ID=gpt-5.4-mini
LLM_API_KEY=your_aihubmix_api_key
LLM_BASE_URL=https://aihubmix.com/v1
```

Embedding calls should stay on DashScope / Aliyun:

```text
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v3
EMBED_API_KEY=your_dashscope_api_key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

`OPENAI_API_KEY` is optional when `LLM_*` and `EMBED_*` are configured. You can leave it empty while testing this block with aihubmix chat and DashScope embeddings.

### Optional: Better PDF Tables With Marker

The default PDF parser is PyMuPDF because it is fast and light. For table-heavy PDFs, use Marker instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[marker]"
```

Then add these values to `.env` and restart the backend:

```text
PDF_PARSER=marker
MARKER_TORCH_DEVICE=cuda
MARKER_FORCE_OCR=false
MARKER_TIMEOUT_SECONDS=1800
```

Marker output is converted into semantic blocks before chunking. Prose is chunked section-first with LangChain header and recursive splitters. Table and figure chunks include `content_type` plus metadata such as `table_footnote`, `table_format`, `image_alt_text`, and `image_path`. Small tables and figures also carry nearby paragraph context into the final chunk text.

Keep OCR off first because it is slower. If a PDF still has badly recognized words after Marker parsing, try `MARKER_FORCE_OCR=true`, restart the backend, remove the old source, upload the PDF again, and approve the new source.

If the full install is slow because of large optional packages such as Chroma and Streamlit, you can still run the focused tests with this smaller dependency set:

```powershell
python -m pip install pytest sqlmodel pydantic pydantic-settings fastapi python-multipart
```

To test semantic indexing/search with DashScope, you also need:

```powershell
python -m pip install openai chromadb
```

## Step-By-Step Tutorial: Test The Existing Function

### 1. Confirm Python Works

```powershell
python --version
```

Expected result: Python 3.11.x.

### 2. Run A Syntax Check

```powershell
python -m compileall -q app tests
```

Expected result: no output and exit code `0`.

### 3. Run The Test Suite

```powershell
python -m pytest -q
```

Expected result:

```text
17 passed
```

These tests verify the current core behavior without needing an OpenAI API key.

### 4. Verify The API App Imports

```powershell
python -c "from app.api.main import app; print(app.title); print(len(app.routes))"
```

Expected result:

```text
Personal Coach Evidence-Based Q&A Assistant
12
```

### 5. Start The Backend

Install `uvicorn` if you used the smaller dependency set:

```powershell
python -m pip install uvicorn
```

Start the backend:

```powershell
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs:

- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 6. Try A No-Evidence Q&A Request

In a second PowerShell window, run:

```powershell
$body = @{
  question = "How much protein should I eat to build muscle?"
  topic_hint = "nutrition"
  debug = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/qa/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Expected behavior:

- The assistant returns an insufficient-evidence answer.
- A `trace_id` is created.
- Debug fields show retrieval and timing information.
- If an embedding provider is configured but the provider request fails, the response stays successful and the debug `semantic_error` field explains the provider issue, such as timeout, quota, or invalid key.

This works without a chat API key because no approved evidence is available, so the assistant refuses to guess.

After you approve a source, the same ask endpoint will retrieve citations from approved evidence. Writing the final prose answer uses the chat model configured by `LLM_API_KEY`, `LLM_MODEL_ID`, and `LLM_BASE_URL` when those values are present. If that chat provider fails, the API returns an evidence fallback instead of crashing or discarding citations.

### 7. Upload A Small Test Source

Create a tiny local test source:

```powershell
New-Item -ItemType Directory -Force tmp | Out-Null
Set-Content -Path tmp\protein_note.txt -Value "Protein supports muscle repair after resistance training. Progressive overload helps strength improve over time."
```

Upload it:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/knowledge/upload" `
  -F "file=@tmp\protein_note.txt" `
  -F "topic_tags=nutrition,strength"
```

If you omit `title`, the app uses the uploaded file name, such as `protein_note`.

Expected behavior:

- The response contains a `source_id`.
- `approval_status` is `pending_review`.
- `parser_status` is `parsed`.
- PDF uploads may include `parser_warnings` when the parser repaired or still detected suspicious line wrapping or character-split text.
- With `PDF_PARSER=marker`, table and figure chunks keep their `content_type` and metadata for retrieval/debug inspection.
- `chunk_count` is at least `1`.

### 7B. Upload The Whole RAG_corpus Folder

The backend can upload every supported file under `RAG_corpus` in one request:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/upload/rag-corpus" `
  -Method Post
```

Expected behavior:

- The response includes `uploaded_count`, `skipped_count`, `uploaded`, and `skipped`.
- Each uploaded source title comes from its file name.
- Each uploaded source tag comes from its folder under `RAG_corpus`, such as `sports_nutrition`.
- Uploaded sources are still `pending_review`; approve them before asking questions that need them.

The Streamlit `Upload` tab uses a file-by-file import path for this same folder, so it can show a progress bar while uploading large corpora.

### 8. Confirm Pending Sources Are Not Searchable

Use the `source_id` from the upload response only for later approval. Pending sources should not be used in answers or search.

```powershell
$searchBody = @{
  query = "protein muscle repair"
  topic_hint = "nutrition"
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/search" `
  -Method Post `
  -ContentType "application/json" `
  -Body $searchBody
```

Expected behavior:

- No approved keyword chunks are returned while the source is pending.
- If no `OPENAI_API_KEY` is configured, semantic search reports a clear configuration error.

### 9. Approve A Source

Approval requires semantic indexing, so set the `EMBED_*` values in `.env` before doing this if you want the source to become fully approved and searchable.

For DashScope / Aliyun embeddings, use:

```text
EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v3
EMBED_API_KEY=your_dashscope_api_key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_BATCH_SIZE=10
```

Restart the backend after editing `.env`.

Without a working embedding API key, approval returns `index_error` by design. This prevents partially indexed material from becoming trusted approved knowledge.
For large PDFs, keep `EMBEDDING_BATCH_SIZE` small enough for your embedding provider. DashScope rejects batches above `10`, so the app caps DashScope requests at `10` even if `.env` is higher. If a provider still rejects a large approval request, try `8` or `4`, restart the backend, and approve the source again.

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/YOUR_SOURCE_ID/approve" `
  -Method Post
```

Replace `YOUR_SOURCE_ID` with the uploaded source ID.

Expected behavior with no working embedding API key:

- `approval_status` is `index_error`.
- The response message mentions the missing or failed embedding provider.

Expected behavior with valid DashScope settings and full dependencies:

- `approval_status` is `approved`.
- The source is indexed into SQLite FTS5 and Chroma.

After approval succeeds, search again:

```powershell
$searchBody = @{
  query = "protein muscle repair"
  topic_hint = "nutrition"
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/search" `
  -Method Post `
  -ContentType "application/json" `
  -Body $searchBody
```

Expected behavior:

- At least one approved chunk appears.
- You should see retrieval methods such as `keyword` or `semantic`.

### 9A. Review Multiple Sources

Approve or reject selected sources in one API request:

```powershell
$reviewBody = @{
  action = "reject"
  source_ids = @("src_first", "src_second")
  review_all = $false
  review_note = "Not useful for this knowledge base."
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/review/batch" `
  -Method Post `
  -ContentType "application/json" `
  -Body $reviewBody
```

Approve all pending or index-error sources:

```powershell
$reviewBody = @{
  action = "approve"
  source_ids = @()
  review_all = $true
  review_note = ""
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/review/batch" `
  -Method Post `
  -ContentType "application/json" `
  -Body $reviewBody
```

Use `action = "reject"` with `review_all = $true` to reject all reviewable sources.

### 9B. Ask With Approved Evidence And Aihubmix Chat

Make sure `.env` has both provider sections:

```text
LLM_MODEL_TYPE=aihubmix
LLM_MODEL_ID=gpt-5.4-mini
LLM_API_KEY=your_aihubmix_api_key
LLM_BASE_URL=https://aihubmix.com/v1

EMBED_MODEL_TYPE=dashscope
EMBED_MODEL_NAME=text-embedding-v3
EMBED_API_KEY=your_dashscope_api_key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Restart the backend after editing `.env`, then ask:

```powershell
$body = @{
  question = "How much protein should I eat to build muscle?"
  topic_hint = "nutrition"
  debug = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/qa/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Expected behavior:

- `semantic_error` is empty or null.
- `source_ids` and `citations` include the approved source.
- The answer is generated by the configured aihubmix chat model.

### 10. Inspect Sources

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/knowledge/sources"
```

Optional status filters:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/knowledge/sources?status=pending_review"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/knowledge/sources?status=approved"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/knowledge/sources?status=index_error"
```

### 10B. Remove Approved Sources

Remove one approved source:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/sources/YOUR_SOURCE_ID" `
  -Method Delete
```

Remove multiple approved sources:

```powershell
$removeBody = @{
  source_ids = @("src_first", "src_second")
  remove_all = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/sources/remove" `
  -Method Post `
  -ContentType "application/json" `
  -Body $removeBody
```

Remove all approved sources:

```powershell
$removeBody = @{
  source_ids = @()
  remove_all = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/sources/remove" `
  -Method Post `
  -ContentType "application/json" `
  -Body $removeBody
```

Removal deletes the approved source row, its chunk rows, matching citation rows, its keyword index rows, its vector index rows, and the copied file under `data/uploads`.

### 10C. Remove Rejected Source Records

Remove historical rejected records:

```powershell
$removeBody = @{
  source_ids = @()
  remove_all = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/knowledge/sources/rejected/remove" `
  -Method Post `
  -ContentType "application/json" `
  -Body $removeBody
```

You can also pass specific rejected `source_ids` with `remove_all = $false`.

### 11. Inspect An Answer Trace

Use a `trace_id` from an `/api/qa/ask` response:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/qa/traces/YOUR_TRACE_ID"
```

Expected behavior:

- You can see the question, retrieved chunk IDs, merged chunk IDs, final context chunk IDs, answer, timings, citations, safety flags, and errors.
- In Streamlit, the `Trace` tab also has a `View retrieved chunks` expander that shows the full stored text for citation chunks, merged chunks, and final chunks.

### 12. Try The Streamlit UI

Install Streamlit if needed:

```powershell
python -m pip install streamlit httpx
```

Start the backend first, then run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

UI tabs:

- `Chat`: ask questions, preview retrieval, and read answers as formatted Markdown with a citation table. The `View retrieved chunks` expander shows the full text for citation chunks, merged chunks, and final chunks, including parse quality warnings when a chunk still looks suspicious.
- `Upload`: upload a source file or import all supported files from `RAG_corpus` with progress.
- `Review`: approve/reject one source, selected sources, or all reviewable sources with progress.
- `Sources`: inspect source status, remove approved sources, and remove rejected history records.
- `Trace`: load a trace by ID and inspect the full text behind citation, merged, and final chunk IDs.


UI: `http://localhost:8501`

Parser improvements only affect newly parsed files. To improve older malformed chunks, remove the old approved source, upload the PDF again, and approve the newly parsed source.

## Run Tests

```powershell
pytest
```
