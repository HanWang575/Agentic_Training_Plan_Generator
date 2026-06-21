"""Minimal Streamlit UI for local review, chat, and trace inspection."""

import json
import sys
import time
from datetime import timedelta
from pathlib import Path

# Streamlit executes this file with `app/ui` on sys.path. Add the project root
# so absolute imports such as `app.core.config` work even without editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import streamlit as st

from app.core.config import get_settings
from app.ui.formatting import answer_to_markdown, citations_to_rows


settings = get_settings()
API_BASE_URL = settings.api_base_url
TRAINING_SPORT_TYPES = [
    "strength",
    "running",
    "other aerobic exercise",
    "other anaerobic exercise",
    "other Mixed Aerobic-Anaerobic Exercise",
]


def _clean_id_input(value: str | None) -> str:
    """Accept IDs pasted raw or with surrounding JSON/string quotes."""

    text = (value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, str):
        return parsed.strip()
    quote_pairs = {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")}
    while len(text) >= 2 and (text[0], text[-1]) in quote_pairs:
        text = text[1:-1].strip()
    return text


def _rerun_app() -> None:
    """Rerun Streamlit after updating session state."""

    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _fetch_memory_items_for_review(user_id: str, status: str = "candidate") -> list[dict]:
    """Load memory items for the review queue UI."""

    params = {"user_id": user_id}
    if status != "any":
        params["status"] = status
    response = httpx.get(
        f"{API_BASE_URL}/api/memory/items",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _candidate_memory_items(items: list[dict]) -> list[dict]:
    """Return only reviewable candidate memory items from a loaded item list."""

    return [item for item in items if item.get("status") == "candidate" and item.get("memory_id")]


def _queue_next_memory_candidate(items: list[dict]) -> None:
    """Prefill the memory-review text input with the next candidate ID."""

    candidates = _candidate_memory_items(items)
    next_id = candidates[0]["memory_id"] if candidates else ""
    st.session_state["memory_review_pending_id"] = next_id


def _fetch_key_proposals_for_review(user_id: str, review_status: str = "pending") -> list[dict]:
    """Load key proposals for the review queue UI."""

    params = {"user_id": user_id}
    if review_status != "any":
        params["review_status"] = review_status
    response = httpx.get(
        f"{API_BASE_URL}/api/memory/key-proposals",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _pending_key_proposals(items: list[dict]) -> list[dict]:
    """Return only reviewable pending key proposals from a loaded proposal list."""

    return [item for item in items if item.get("review_status") == "pending" and item.get("proposal_id")]


def _queue_next_key_proposal(items: list[dict]) -> None:
    """Prefill the key-proposal review text input with the next pending proposal ID."""

    proposals = _pending_key_proposals(items)
    next_id = proposals[0]["proposal_id"] if proposals else ""
    st.session_state["key_proposal_review_pending_id"] = next_id


def _sync_xunji_with_progress(
    user_id: str,
    start_date,
    end_date,
    include_full_data: bool,
    force_refresh: bool,
    chunk_days: int = 7,
) -> dict:
    """Import Xunji logs in small date chunks so long ranges do not timeout."""

    if end_date < start_date:
        return {
            "user_id": user_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days_requested": 0,
            "imports": [],
            "errors": [{"datestr": "", "error": "End date must be on or after start date."}],
        }

    total_days = (end_date - start_date).days + 1
    safe_chunk_days = max(1, min(31, int(chunk_days)))
    progress = st.progress(0)
    status = st.empty()
    chunks = []
    imports = []
    errors = []
    completed_days = 0
    current = start_date

    while current <= end_date:
        chunk_end = min(current + timedelta(days=safe_chunk_days - 1), end_date)
        chunk_count = (chunk_end - current).days + 1
        status.write(
            f"Importing Xunji logs {current.isoformat()} to {chunk_end.isoformat()} "
            f"({completed_days + 1}-{completed_days + chunk_count} of {total_days} days)"
        )
        try:
            response = httpx.post(
                f"{API_BASE_URL}/api/memory/imports/xunji/sync",
                json={
                    "user_id": user_id,
                    "start_date": current.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "include_full_data": include_full_data,
                    "force_refresh": force_refresh,
                },
                timeout=max(180, safe_chunk_days * 90),
            )
            response.raise_for_status()
            payload = response.json()
            chunks.append(payload)
            imports.extend(payload.get("imports", []))
            errors.extend(payload.get("errors", []))
        except Exception as exc:
            errors.append(
                {
                    "datestr": f"{current.isoformat()}..{chunk_end.isoformat()}",
                    "error": str(exc),
                }
            )
        completed_days += chunk_count
        progress.progress(min(1.0, completed_days / total_days))
        current = chunk_end + timedelta(days=1)

    imported_sessions = sum(item.get("sessions_imported", 0) for item in imports)
    skipped_duplicates = sum(1 for item in imports if item.get("skipped_duplicate"))
    status.write(
        f"Xunji import finished: {imported_sessions} session(s) imported, "
        f"{skipped_duplicates} duplicate day(s) skipped, {len(errors)} error(s)."
    )
    return {
        "user_id": user_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "days_requested": total_days,
        "chunk_days": safe_chunk_days,
        "sessions_imported": imported_sessions,
        "skipped_duplicate_imports": skipped_duplicates,
        "imports": imports,
        "errors": errors,
        "chunks": chunks,
    }


def _source_label(source: dict) -> str:
    """Return compact source label for selection widgets."""

    return f"{source['title']} ({source['source_id']})"


def _review_sources_with_progress(source_ids: list[str], action: str, review_note: str | None) -> dict:
    """Approve or reject sources one at a time so Streamlit can show progress."""

    progress = st.progress(0)
    status = st.empty()
    results = []
    failed = []
    total = len(source_ids)
    for index, source_id in enumerate(source_ids, start=1):
        status.write(f"{action.title()} {index} of {total}: {source_id}")
        try:
            response = httpx.post(
                f"{API_BASE_URL}/api/knowledge/{source_id}/{action}",
                params={"review_note": review_note or ""},
                timeout=600 if action == "approve" else 120,
            )
            response.raise_for_status()
            results.append(response.json())
        except Exception as exc:
            failed.append({"source_id": source_id, "error": str(exc)})
        progress.progress(index / total)
    status.write(f"{action.title()} complete: {len(results)} succeeded, {len(failed)} failed.")
    return {"results": results, "failed": failed}


def _upload_rag_corpus_with_progress(marker_force_ocr: bool | None = None) -> dict:
    """Upload RAG_corpus files one at a time with visible progress."""

    plan_response = httpx.get(f"{API_BASE_URL}/api/knowledge/upload/rag-corpus/plan", timeout=60)
    plan_response.raise_for_status()
    plan = plan_response.json()
    files = plan.get("files", [])
    progress = st.progress(0)
    status = st.empty()
    uploaded = []
    failed = []
    total = len(files)
    if total == 0:
        status.write("No supported RAG_corpus files found.")
        return {"uploaded": uploaded, "failed": failed, "skipped": plan.get("skipped", [])}
    for index, item in enumerate(files, start=1):
        status.write(f"Uploading {index} of {total}: {item['relative_path']}")
        try:
            response = httpx.post(
                f"{API_BASE_URL}/api/knowledge/upload/rag-corpus/file",
                params={
                    "relative_path": item["relative_path"],
                    "marker_force_ocr": marker_force_ocr,
                },
                timeout=300,
            )
            response.raise_for_status()
            uploaded.append(response.json())
        except Exception as exc:
            failed.append({"path": item["relative_path"], "error": str(exc)})
        progress.progress(index / total)
    status.write(f"Upload complete: {len(uploaded)} uploaded, {len(failed)} failed.")
    return {"uploaded": uploaded, "failed": failed, "skipped": plan.get("skipped", [])}


def _poll_source_until_parsed(source_id: str, poll_seconds: float = 1.0) -> dict:
    """Poll one source summary until parsing finishes."""

    progress = st.progress(0)
    status = st.empty()
    details = st.empty()
    while True:
        response = httpx.get(f"{API_BASE_URL}/api/knowledge/sources/{source_id}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        parser_progress = float(payload.get("parser_progress", 0.0) or 0.0)
        progress.progress(min(1.0, max(0.0, parser_progress)))
        status.write(
            f"{payload.get('parser_status', 'pending').title()}: "
            f"{payload.get('parser_progress_message') or 'Working...'}"
        )
        details.caption(
            f"Source: {payload.get('title', '')} | {payload.get('source_id', '')} | "
            f"Chunks so far: {payload.get('chunk_count', 0)}"
        )
        if payload.get("parser_status") in {"parsed", "parse_error"}:
            return payload
        time.sleep(poll_seconds)


def _upload_source_with_progress(
    uploaded_file,
    title: str,
    topic_tags: str,
    marker_force_ocr: bool | None = None,
) -> dict:
    """Upload one source through the async endpoint and show parse progress."""

    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    data = {"title": title, "topic_tags": topic_tags}
    if marker_force_ocr is not None:
        data["marker_force_ocr"] = "true" if marker_force_ocr else "false"
    response = httpx.post(
        f"{API_BASE_URL}/api/knowledge/upload/async",
        files=files,
        data=data,
        timeout=120,
    )
    response.raise_for_status()
    initial_payload = response.json()
    st.info(f"Upload accepted: {initial_payload['source_id']}")
    return _poll_source_until_parsed(initial_payload["source_id"])


def _unique_ids(chunk_ids: list[str]) -> list[str]:
    """Return chunk IDs without blanks or duplicates, preserving input order.

    Parameters:
        chunk_ids: list with shape `[candidate_count]`.
    Returns:
        list[str]: cleaned IDs with shape `[unique_chunk_count]`.
    """

    return list(dict.fromkeys(chunk_id for chunk_id in chunk_ids if chunk_id))


def _citation_chunk_ids(citations: list[dict]) -> list[str]:
    """Extract citation chunk IDs from API citation objects.

    Parameters:
        citations: list with shape `[citation_count]`.
    Returns:
        list[str]: citation chunk IDs in display order.
    """

    return _unique_ids([citation.get("chunk_id", "") for citation in citations])


def _fetch_chunks_by_id(chunk_ids: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Fetch full chunk text from the backend.

    Parameters:
        chunk_ids: list with shape `[chunk_count]`.
    Returns:
        tuple:
            dict keyed by `chunk_id`, where each value is a chunk detail object.
            list of requested chunk IDs that were not found by the backend.
    """

    requested_ids = _unique_ids(chunk_ids)
    if not requested_ids:
        return {}, []
    response = httpx.post(
        f"{API_BASE_URL}/api/knowledge/chunks",
        json={"chunk_ids": requested_ids},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in payload.get("chunks", [])}
    return chunks_by_id, payload.get("missing_chunk_ids", [])


def _render_chunk_group(title: str, chunk_ids: list[str], chunks_by_id: dict[str, dict], key_prefix: str) -> None:
    """Render one labeled group of retrieved chunks.

    Parameters:
        title: group label shown in Streamlit.
        chunk_ids: list with shape `[group_chunk_count]`.
        chunks_by_id: lookup table returned by `_fetch_chunks_by_id`.
        key_prefix: stable Streamlit widget key prefix for this group.
    Returns:
        None. Streamlit renders the text areas directly.
    """

    cleaned_ids = _unique_ids(chunk_ids)
    st.markdown(f"#### {title}")
    if not cleaned_ids:
        st.caption("No chunks in this group.")
        return
    for index, chunk_id in enumerate(cleaned_ids, start=1):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            st.warning(f"{chunk_id} was not found in local storage.")
            continue
        location = chunk.get("section_heading") or (
            f"page {chunk['page_number']}" if chunk.get("page_number") is not None else "section unavailable"
        )
        st.caption(
            f"{index}. {chunk.get('title', '')} | {chunk_id} | {location} | "
            f"type: {chunk.get('content_type', 'plain_text')} | "
            f"tokens: {chunk.get('token_count', 0)} | status: {chunk.get('approval_status', '')}"
        )
        if chunk.get("content_metadata"):
            st.json(chunk["content_metadata"])
        if chunk.get("quality_warnings"):
            st.warning("Parse quality warning: " + "; ".join(chunk["quality_warnings"]))
        st.text_area(
            "Full text",
            value=chunk.get("text", ""),
            height=220,
            disabled=True,
            key=f"{key_prefix}_{chunk_id}",
            label_visibility="collapsed",
        )


def _render_retrieved_chunks_viewer(payload: dict, source: str) -> None:
    """Show full text for citations, merged chunks, and final chunks.

    Parameters:
        payload: response from `/api/qa/ask` or `/api/qa/traces/{trace_id}`.
        source: stable label used only to keep Streamlit widget keys unique.
    Returns:
        None. Streamlit renders one expander.
    """

    debug = payload.get("debug") or {}
    citation_ids = _citation_chunk_ids(payload.get("citations", []))
    merged_ids = payload.get("merged_chunk_ids") or debug.get("merged_chunk_ids") or []
    final_ids = payload.get("final_chunk_ids") or debug.get("final_chunk_ids") or []
    all_ids = _unique_ids(citation_ids + merged_ids + final_ids)

    with st.expander("View retrieved chunks"):
        if not all_ids:
            st.caption("No retrieved chunk IDs were returned.")
            return
        try:
            chunks_by_id, missing_ids = _fetch_chunks_by_id(all_ids)
        except Exception as exc:
            st.error(f"Could not load chunk text: {exc}")
            return
        if missing_ids:
            st.warning(f"Missing chunks: {', '.join(missing_ids)}")
        _render_chunk_group("Citations", citation_ids, chunks_by_id, f"{source}_citations")
        _render_chunk_group("Merged chunks", merged_ids, chunks_by_id, f"{source}_merged")
        _render_chunk_group("Final chunks", final_ids, chunks_by_id, f"{source}_final")


def _render_query_planner_view(payload: dict, source: str) -> None:
    """Show how the query planner rewrote the user's question.

    Parameters:
        payload: response from `/api/qa/ask` or `/api/qa/traces/{trace_id}`.
        source: stable label used only to keep Streamlit widget keys unique.
    Returns:
        None. Streamlit renders one expander when planner data exists.
    """

    debug = payload.get("debug") or {}
    planner_used = payload.get("planner_used") or debug.get("planner_used")
    clean_question = payload.get("clean_question") or debug.get("clean_question")
    clarified_question = payload.get("clarified_question") or debug.get("clarified_question")
    search_query = payload.get("search_query") or debug.get("search_query")
    query_terms = payload.get("query_terms") or debug.get("query_terms") or []
    planner_notes = payload.get("planner_notes") or debug.get("planner_notes") or []

    if not any([planner_used, clean_question, clarified_question, search_query, query_terms, planner_notes]):
        return

    with st.expander("Query planner"):
        st.caption(f"Planner used: {planner_used or 'unknown'}")
        if clean_question:
            st.text_area(
                "Clean question",
                value=clean_question,
                height=80,
                disabled=True,
                key=f"{source}_planner_clean",
            )
        if clarified_question:
            st.text_area(
                "Clarified question",
                value=clarified_question,
                height=110,
                disabled=True,
                key=f"{source}_planner_clarified",
            )
        if search_query:
            st.text_area(
                "Search query",
                value=search_query,
                height=90,
                disabled=True,
                key=f"{source}_planner_search_query",
            )
        if query_terms:
            st.write("Query terms")
            st.code("\n".join(f"- {term}" for term in query_terms), language="text")
        if planner_notes:
            st.write("Planner notes")
            st.code("\n".join(f"- {note}" for note in planner_notes), language="text")


def _render_qa_response(payload: dict, show_debug: bool) -> None:
    """Render a Q&A API response as readable Streamlit content."""

    st.markdown(answer_to_markdown(payload.get("answer", "")))
    if payload.get("insufficient_evidence"):
        st.warning("The assistant marked this answer as insufficiently supported by approved evidence.")
    if payload.get("safety_note"):
        st.info(payload["safety_note"])

    citations = citations_to_rows(payload.get("citations", []))
    if citations:
        st.subheader("Citations")
        st.dataframe(citations, hide_index=True, use_container_width=True)

    trace_id = payload.get("trace_id")
    if trace_id:
        st.caption(f"Trace ID: {trace_id}")
    conversation_id = payload.get("conversation_id")
    if conversation_id:
        st.caption(f"Memory conversation ID: {conversation_id}")
    memory_context = (payload.get("debug") or {}).get("memory_context") or {}
    for warning in memory_context.get("budget_warnings", []):
        st.warning(warning.get("message", str(warning)))

    _render_query_planner_view(payload, source=f"chat_{trace_id or 'latest'}")
    _render_retrieved_chunks_viewer(payload, source=f"chat_{trace_id or 'latest'}")

    if show_debug and payload.get("debug"):
        with st.expander("Debug trace"):
            st.json(payload["debug"])
    with st.expander("Raw API response"):
        st.json(payload)


def _render_trace_response(payload: dict) -> None:
    """Render a stored trace and its retrieved chunk text.

    Parameters:
        payload: response from `/api/qa/traces/{trace_id}`.
    Returns:
        None. Streamlit renders the JSON and chunk viewer directly.
    """

    _render_query_planner_view(payload, source=f"trace_{payload.get('trace_id', 'lookup')}")
    st.json(payload)
    _render_retrieved_chunks_viewer(payload, source=f"trace_{payload.get('trace_id', 'lookup')}")


st.set_page_config(page_title="Personal Coach AI Agent", layout="wide")
st.title("Personal Coach AI Agent")

tabs = st.tabs(["Chat", "Upload", "Review", "Sources", "Trace", "Training Log", "Memory", "Program Generator"])


with tabs[0]:
    st.subheader("Ask")
    question = st.text_area("Question", height=120)
    topic_hint = st.text_input("Topic hint")
    qa_user_id = _clean_id_input(st.text_input("Memory user ID", value=""))
    if (
        not st.session_state.get("qa_memory_conversation_id")
        and st.session_state.get("last_qa_memory_conversation_id")
    ):
        st.session_state["qa_memory_conversation_id"] = st.session_state[
            "last_qa_memory_conversation_id"
        ]
    qa_conversation_id = _clean_id_input(
        st.text_input("Memory conversation ID", key="qa_memory_conversation_id")
    )
    st.caption("Leave blank to auto-create a memory conversation when a memory user ID is provided.")
    include_memory = st.checkbox("Include memory context", value=False)
    memory_debug = st.checkbox("Show memory debug", value=False)
    debug = st.checkbox("Show debug trace", value=True)
    if st.button("Preview retrieval"):
        response = httpx.post(
            f"{API_BASE_URL}/api/knowledge/search",
            json={"query": question, "topic_hint": topic_hint or None, "top_k": 10},
            timeout=60,
        )
        st.json(response.json())
    if st.button("Ask", type="primary"):
        response = httpx.post(
            f"{API_BASE_URL}/api/qa/ask",
            json={
                "question": question,
                "topic_hint": topic_hint or None,
                "debug": True,
                "user_id": qa_user_id or None,
                "conversation_id": qa_conversation_id or None,
                "include_memory": include_memory,
                "memory_debug": memory_debug,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        resolved_conversation_id = payload.get("conversation_id")
        if resolved_conversation_id:
            st.session_state["last_qa_memory_conversation_id"] = resolved_conversation_id
            if not qa_conversation_id:
                st.info(f"Created memory conversation: {resolved_conversation_id}")
        _render_qa_response(payload, show_debug=debug)


with tabs[1]:
    st.subheader("Upload Source")
    uploaded_file = st.file_uploader("Source file", type=["pdf", "epub", "md", "txt", "json"])
    upload_title = st.text_input("Title", placeholder="Auto from file name")
    upload_tags = st.text_input("Topic tags, comma separated", placeholder="Optional")
    upload_marker_ocr = st.checkbox("Use Marker OCR preprocessing for PDF upload", value=False)
    if st.button("Upload") and uploaded_file is not None:
        st.json(_upload_source_with_progress(uploaded_file, upload_title, upload_tags, upload_marker_ocr))
    if st.button("Upload RAG_corpus"):
        st.json(_upload_rag_corpus_with_progress(upload_marker_ocr))


with tabs[2]:
    st.subheader("Source Review")
    review_sources = []
    for review_status in ("pending_review", "index_error"):
        review_sources.extend(
            httpx.get(
                f"{API_BASE_URL}/api/knowledge/sources",
                params={"status": review_status},
                timeout=30,
            ).json()
        )
    review_by_id = {source["source_id"]: source for source in review_sources}
    selected_review_ids = st.multiselect(
        "Sources to review",
        options=list(review_by_id),
        format_func=lambda source_id: _source_label(review_by_id[source_id]),
    )
    batch_note = st.text_input("Batch review note")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Approve selected") and selected_review_ids:
        st.json(_review_sources_with_progress(selected_review_ids, "approve", batch_note))
    if col2.button("Reject selected") and selected_review_ids:
        st.json(_review_sources_with_progress(selected_review_ids, "reject", batch_note))
    all_review_ids = list(review_by_id)
    if col3.button("Approve all") and all_review_ids:
        st.json(_review_sources_with_progress(all_review_ids, "approve", batch_note))
    if col4.button("Reject all") and all_review_ids:
        st.json(_review_sources_with_progress(all_review_ids, "reject", batch_note))
    for source in review_sources:
        with st.expander(f"{source['title']} ({source['source_id']})"):
            st.json(source)
            note = st.text_input("Review note", key=f"note_{source['source_id']}")
            col1, col2 = st.columns(2)
            if col1.button("Approve", key=f"approve_{source['source_id']}"):
                response = httpx.post(
                    f"{API_BASE_URL}/api/knowledge/{source['source_id']}/approve",
                    params={"review_note": note},
                    timeout=600,
                )
                st.json(response.json())
            if col2.button("Reject", key=f"reject_{source['source_id']}"):
                response = httpx.post(
                    f"{API_BASE_URL}/api/knowledge/{source['source_id']}/reject",
                    params={"review_note": note},
                    timeout=120,
                )
                st.json(response.json())


with tabs[3]:
    st.subheader("Sources")
    status = st.selectbox("Status", ["", "approved", "pending_review", "rejected", "parse_error", "index_error"])
    params = {"status": status} if status else {}
    response = httpx.get(f"{API_BASE_URL}/api/knowledge/sources", params=params, timeout=30)
    sources = response.json()
    st.json(sources)

    approved_sources = httpx.get(
        f"{API_BASE_URL}/api/knowledge/sources",
        params={"status": "approved"},
        timeout=30,
    ).json()
    approved_by_id = {source["source_id"]: source for source in approved_sources}
    selected_source_ids = st.multiselect(
        "Approved sources to remove",
        options=list(approved_by_id),
        format_func=lambda source_id: _source_label(approved_by_id[source_id]),
    )
    col1, col2 = st.columns(2)
    if col1.button("Remove selected") and selected_source_ids:
        response = httpx.post(
            f"{API_BASE_URL}/api/knowledge/sources/remove",
            json={"source_ids": selected_source_ids, "remove_all": False},
            timeout=300,
        )
        st.json(response.json())
    if col2.button("Remove all approved"):
        response = httpx.post(
            f"{API_BASE_URL}/api/knowledge/sources/remove",
            json={"source_ids": [], "remove_all": True},
            timeout=600,
        )
        st.json(response.json())

    rejected_sources = httpx.get(
        f"{API_BASE_URL}/api/knowledge/sources",
        params={"status": "rejected"},
        timeout=30,
    ).json()
    rejected_by_id = {source["source_id"]: source for source in rejected_sources}
    selected_rejected_ids = st.multiselect(
        "Rejected records to remove",
        options=list(rejected_by_id),
        format_func=lambda source_id: _source_label(rejected_by_id[source_id]),
    )
    col3, col4 = st.columns(2)
    if col3.button("Remove selected rejected") and selected_rejected_ids:
        response = httpx.post(
            f"{API_BASE_URL}/api/knowledge/sources/rejected/remove",
            json={"source_ids": selected_rejected_ids, "remove_all": False},
            timeout=300,
        )
        st.json(response.json())
    if col4.button("Remove all rejected"):
        response = httpx.post(
            f"{API_BASE_URL}/api/knowledge/sources/rejected/remove",
            json={"source_ids": [], "remove_all": True},
            timeout=600,
        )
        st.json(response.json())


with tabs[4]:
    st.subheader("Trace Lookup")
    trace_id = _clean_id_input(st.text_input("Trace ID"))
    if st.button("Load trace") and trace_id:
        response = httpx.get(f"{API_BASE_URL}/api/qa/traces/{trace_id}", timeout=30)
        response.raise_for_status()
        _render_trace_response(response.json())


with tabs[5]:
    st.subheader("Historical Training Log")
    training_user_id = _clean_id_input(st.text_input("Training log user ID", value="usr_local"))

    st.markdown("#### Import From Xunji")
    sync_col1, sync_col2 = st.columns(2)
    xunji_start_date = sync_col1.date_input("Xunji import start date")
    xunji_end_date = sync_col2.date_input("Xunji import end date")
    xunji_include_full = st.checkbox("Include full Xunji data", value=True)
    xunji_force_refresh = st.checkbox("Force Xunji cache refresh", value=False)
    xunji_chunk_days = st.number_input("Import chunk size in days", min_value=1, max_value=31, value=7, step=1)
    if st.button("Import Xunji logs"):
        st.json(
            _sync_xunji_with_progress(
                user_id=training_user_id,
                start_date=xunji_start_date,
                end_date=xunji_end_date,
                include_full_data=xunji_include_full,
                force_refresh=xunji_force_refresh,
                chunk_days=int(xunji_chunk_days),
            )
        )

    st.markdown("#### Training Sessions")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    session_start_date = filter_col1.date_input("Session start date")
    session_end_date = filter_col2.date_input("Session end date")
    session_sport_type = filter_col3.selectbox("Sport type", ["any"] + TRAINING_SPORT_TYPES)
    session_limit = filter_col4.number_input("Session limit", min_value=1, max_value=1000, value=200, step=1)
    if st.button("Load training sessions"):
        params = {
            "user_id": training_user_id,
            "start_date": session_start_date.isoformat(),
            "end_date": session_end_date.isoformat(),
            "limit": int(session_limit),
        }
        if session_sport_type != "any":
            params["sport_type"] = session_sport_type
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/training-log/sessions",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        sessions = payload.get("sessions", [])
        st.caption(f"{payload.get('session_count', 0)} session(s) matched.")
        if sessions:
            rows = [
                {
                    "session_id": item.get("session_id"),
                    "date": item.get("session_date"),
                    "title": item.get("title"),
                    "sport_type": item.get("sport_type"),
                    "duration_seconds": item.get("duration_seconds"),
                    "rpe": item.get("rpe"),
                    "notes": item.get("notes"),
                }
                for item in sessions
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)
        st.json(payload)


with tabs[6]:
    st.subheader("User Memory")
    memory_user_id = _clean_id_input(st.text_input("User ID", value="usr_local"))
    display_name = st.text_input("Display name")
    timezone_name = st.text_input("Timezone", value="UTC")
    delete_user_confirm = st.checkbox("Confirm delete current user and all memory rows", value=False)
    user_col1, user_col2, user_col3 = st.columns(3)
    if user_col1.button("Create or update user"):
        response = httpx.post(
            f"{API_BASE_URL}/api/memory/users",
            json={
                "user_id": memory_user_id,
                "display_name": display_name or None,
                "timezone": timezone_name or "UTC",
            },
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    if user_col2.button("Show existing user IDs"):
        response = httpx.get(f"{API_BASE_URL}/api/memory/users", timeout=30)
        response.raise_for_status()
        users = response.json()
        st.json([user["user_id"] for user in users])
    if user_col3.button("Delete current user and memories"):
        if not delete_user_confirm:
            st.warning("Check the confirmation box before deleting the current user.")
        else:
            response = httpx.delete(f"{API_BASE_URL}/api/memory/users/{memory_user_id}", timeout=30)
            response.raise_for_status()
            st.json(response.json())

    st.markdown("#### Conversations")
    conversation_id_for_action = _clean_id_input(st.text_input("Conversation ID"))
    conversation_close_reason = st.text_input("Conversation close reason", value="user_closed")
    conversation_extract_candidates = st.checkbox("Extract candidates when closing conversation", value=True)
    conversation_col1, conversation_col2, conversation_col3, conversation_col4 = st.columns(4)
    if conversation_col1.button("Show existing conversations"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/conversations",
            params={"user_id": memory_user_id},
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    if conversation_col2.button("Show conversation content"):
        if not conversation_id_for_action:
            st.warning("Enter a conversation ID first.")
        else:
            response = httpx.get(
                f"{API_BASE_URL}/api/memory/conversations/{conversation_id_for_action}",
                timeout=30,
            )
            response.raise_for_status()
            st.json(response.json())
    if conversation_col3.button("Delete conversation"):
        if not conversation_id_for_action:
            st.warning("Enter a conversation ID first.")
        else:
            response = httpx.delete(
                f"{API_BASE_URL}/api/memory/conversations/{conversation_id_for_action}",
                timeout=30,
            )
            response.raise_for_status()
            st.json(response.json())
    if conversation_col4.button("Close conversation"):
        if not conversation_id_for_action:
            st.warning("Enter a conversation ID first.")
        else:
            response = httpx.post(
                f"{API_BASE_URL}/api/memory/conversations/{conversation_id_for_action}/close",
                json={
                    "closure_reason": conversation_close_reason or "user_closed",
                    "extract_candidates": conversation_extract_candidates,
                },
                timeout=30,
            )
            response.raise_for_status()
            st.json(response.json())

    st.markdown("#### Profile")
    profile_json = st.text_area(
        "Profile JSON",
        value='{"primary_training_goal": "build strength"}',
        height=100,
    )
    profile_extension_mode = st.checkbox("Profile extension mode", value=False)
    if st.button("Save profile fields"):
        response = httpx.patch(
            f"{API_BASE_URL}/api/memory/profile/{memory_user_id}",
            json={
                "fields": json.loads(profile_json or "{}"),
                "extension_mode": profile_extension_mode,
            },
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    profile_field_id_to_delete = _clean_id_input(st.text_input("Profile field ID to delete"))
    profile_col1, profile_col2, profile_col3 = st.columns(3)
    if profile_col1.button("Load profile"):
        response = httpx.get(f"{API_BASE_URL}/api/memory/profile/{memory_user_id}", timeout=30)
        response.raise_for_status()
        st.json(response.json())
    if profile_col2.button("Show profile keys"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/keys/profile",
            params={"user_id": memory_user_id},
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    if profile_col3.button("Delete profile entry"):
        if not profile_field_id_to_delete:
            st.warning("Enter a profile field ID first.")
        else:
            response = httpx.delete(
                f"{API_BASE_URL}/api/memory/profile-fields/{profile_field_id_to_delete}",
                timeout=30,
            )
            response.raise_for_status()
            st.json(response.json())

    st.markdown("#### Recent Status")
    status_json = st.text_area(
        "Recent status JSON",
        value='{"sleep_duration_hours": 7}',
        height=100,
    )
    status_extension_mode = st.checkbox("Recent status extension mode", value=False)
    if st.button("Save recent status"):
        response = httpx.patch(
            f"{API_BASE_URL}/api/memory/recent-status/users/{memory_user_id}",
            json={
                "statuses": json.loads(status_json or "{}"),
                "extension_mode": status_extension_mode,
            },
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    recent_status_id_for_action = _clean_id_input(st.text_input("Recent status ID"))
    recent_status_new_valid_until = st.text_input("New recent status valid_until", placeholder="2026-06-17T12:00:00Z")
    status_col1, status_col2, status_col3, status_col4, status_col5, status_col6, status_col7 = st.columns(7)
    if status_col1.button("Load current recent status"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/recent-status/current",
            params={"user_id": memory_user_id},
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    if status_col2.button("Show recent status keys"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/keys/recent-status",
            params={"user_id": memory_user_id},
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    if status_col3.button("Load expiring status"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/recent-status/expiring",
            params={"user_id": memory_user_id},
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    if status_col4.button("Delete recent status entry"):
        if not recent_status_id_for_action:
            st.warning("Enter a recent status ID first.")
        else:
            response = httpx.delete(
                f"{API_BASE_URL}/api/memory/recent-status/{recent_status_id_for_action}",
                timeout=30,
            )
            response.raise_for_status()
            st.json(response.json())
    if status_col5.button("Expire recent status"):
        if not recent_status_id_for_action:
            st.warning("Enter a recent status ID first.")
        else:
            response = httpx.post(
                f"{API_BASE_URL}/api/memory/recent-status/{recent_status_id_for_action}/expire",
                json={"reason": "user_expired_from_memory_tab"},
                timeout=30,
            )
            response.raise_for_status()
            st.json(response.json())
    if status_col6.button("Renew recent status"):
        if not recent_status_id_for_action or not recent_status_new_valid_until:
            st.warning("Enter a recent status ID and new valid_until first.")
        else:
            response = httpx.patch(
                f"{API_BASE_URL}/api/memory/recent-status/{recent_status_id_for_action}/valid-until",
                json={"valid_until": recent_status_new_valid_until},
                timeout=30,
            )
            response.raise_for_status()
            st.json(response.json())
    if status_col7.button("Load status audit"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/recent-status/current",
            params={"user_id": memory_user_id, "include_expired": True},
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())

    st.markdown("#### Context")
    memory_query = st.text_input("Memory query", value="How should I train today?")
    memory_conversation_id = _clean_id_input(st.text_input("Conversation ID", placeholder="Optional"))
    context_col1, context_col2, context_col3 = st.columns(3)
    if context_col1.button("Build memory context"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/context",
            params={
                "user_id": memory_user_id,
                "query": memory_query,
                "conversation_id": memory_conversation_id or None,
                "memory_debug": True,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        for warning in payload.get("budget_warnings", []):
            st.warning(warning.get("message", str(warning)))
        st.json(payload)
    if context_col2.button("Search episodes"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/episodes/search",
            params={"user_id": memory_user_id, "query": memory_query},
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())
    if context_col3.button("List all episodes"):
        response = httpx.get(
            f"{API_BASE_URL}/api/memory/episodes/search",
            params={"user_id": memory_user_id, "query": "", "top_k": 100},
            timeout=30,
        )
        response.raise_for_status()
        st.json(response.json())

    st.markdown("#### Review Queues")
    memory_item_status = st.selectbox(
        "Memory item status",
        ["any", "candidate", "approved", "rejected"],
    )
    if "memory_review_items" not in st.session_state:
        st.session_state["memory_review_items"] = []
    if "memory_review_last_result" not in st.session_state:
        st.session_state["memory_review_last_result"] = None
    if "memory_item_review_id_input" not in st.session_state:
        st.session_state["memory_item_review_id_input"] = ""
    if "memory_review_pending_id" in st.session_state:
        st.session_state["memory_item_review_id_input"] = st.session_state.pop("memory_review_pending_id")

    memory_item_review_id = _clean_id_input(
        st.text_input("Memory item ID to review", key="memory_item_review_id_input")
    )
    memory_item_review_note = st.text_input("Memory item review note")
    col1, col2, col3 = st.columns(3)
    if col1.button("Load memory items"):
        items = _fetch_memory_items_for_review(memory_user_id, memory_item_status)
        st.session_state["memory_review_items"] = items
        st.session_state["memory_review_last_result"] = None
        _queue_next_memory_candidate(items)
        _rerun_app()
    if col2.button("Approve memory item"):
        if not memory_item_review_id:
            st.warning("Enter a candidate memory item ID first.")
        else:
            response = httpx.post(
                f"{API_BASE_URL}/api/memory/items/{memory_item_review_id}/approve",
                json={"review_note": memory_item_review_note or None},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["memory_review_last_result"] = response.json()
            items = _fetch_memory_items_for_review(memory_user_id, "candidate")
            st.session_state["memory_review_items"] = items
            _queue_next_memory_candidate(items)
            _rerun_app()
    if col3.button("Reject memory item"):
        if not memory_item_review_id:
            st.warning("Enter a candidate memory item ID first.")
        else:
            response = httpx.post(
                f"{API_BASE_URL}/api/memory/items/{memory_item_review_id}/reject",
                json={"review_note": memory_item_review_note or None},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["memory_review_last_result"] = response.json()
            items = _fetch_memory_items_for_review(memory_user_id, "candidate")
            st.session_state["memory_review_items"] = items
            _queue_next_memory_candidate(items)
            _rerun_app()

    loaded_memory_items = st.session_state.get("memory_review_items", [])
    candidate_items = _candidate_memory_items(loaded_memory_items)
    current_candidate = next(
        (item for item in candidate_items if item.get("memory_id") == memory_item_review_id),
        candidate_items[0] if candidate_items else None,
    )
    if st.session_state.get("memory_review_last_result"):
        st.markdown("##### Last review result")
        st.json(st.session_state["memory_review_last_result"])
    if current_candidate:
        st.markdown("##### Current candidate")
        st.caption(
            f"{len(candidate_items)} candidate item(s) loaded. "
            "Approve or reject this item to advance to the next candidate."
        )
        st.json(current_candidate)
    elif loaded_memory_items:
        st.info("No candidate memory items are loaded for review.")
        st.json(loaded_memory_items)

    key_proposal_review_status = st.selectbox(
        "Key proposal status",
        ["any", "pending", "approved", "rejected"],
    )
    if "key_proposal_review_items" not in st.session_state:
        st.session_state["key_proposal_review_items"] = []
    if "key_proposal_review_last_result" not in st.session_state:
        st.session_state["key_proposal_review_last_result"] = None
    if "key_proposal_review_id_input" not in st.session_state:
        st.session_state["key_proposal_review_id_input"] = ""
    if "key_proposal_review_pending_id" in st.session_state:
        st.session_state["key_proposal_review_id_input"] = st.session_state.pop("key_proposal_review_pending_id")

    key_proposal_review_id = _clean_id_input(
        st.text_input("Key proposal ID to review", key="key_proposal_review_id_input")
    )
    proposal_col1, proposal_col2, proposal_col3 = st.columns(3)
    if proposal_col1.button("Load key proposals"):
        proposals = _fetch_key_proposals_for_review(memory_user_id, key_proposal_review_status)
        st.session_state["key_proposal_review_items"] = proposals
        st.session_state["key_proposal_review_last_result"] = None
        _queue_next_key_proposal(proposals)
        _rerun_app()
    if proposal_col2.button("Approve key proposal"):
        if not key_proposal_review_id:
            st.warning("Enter a key proposal ID first.")
        else:
            response = httpx.post(
                f"{API_BASE_URL}/api/memory/key-proposals/{key_proposal_review_id}/approve",
                json={},
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["key_proposal_review_last_result"] = response.json()
            proposals = _fetch_key_proposals_for_review(memory_user_id, "pending")
            st.session_state["key_proposal_review_items"] = proposals
            _queue_next_key_proposal(proposals)
            _rerun_app()
    if proposal_col3.button("Reject key proposal"):
        if not key_proposal_review_id:
            st.warning("Enter a key proposal ID first.")
        else:
            response = httpx.post(
                f"{API_BASE_URL}/api/memory/key-proposals/{key_proposal_review_id}/reject",
                timeout=30,
            )
            response.raise_for_status()
            st.session_state["key_proposal_review_last_result"] = response.json()
            proposals = _fetch_key_proposals_for_review(memory_user_id, "pending")
            st.session_state["key_proposal_review_items"] = proposals
            _queue_next_key_proposal(proposals)
            _rerun_app()

    loaded_key_proposals = st.session_state.get("key_proposal_review_items", [])
    pending_key_proposals = _pending_key_proposals(loaded_key_proposals)
    current_key_proposal = next(
        (item for item in pending_key_proposals if item.get("proposal_id") == key_proposal_review_id),
        pending_key_proposals[0] if pending_key_proposals else None,
    )
    if st.session_state.get("key_proposal_review_last_result"):
        st.markdown("##### Last key proposal review result")
        st.json(st.session_state["key_proposal_review_last_result"])
    if current_key_proposal:
        st.markdown("##### Current key proposal")
        st.caption(
            f"{len(pending_key_proposals)} pending proposal(s) loaded. "
            "Approve or reject this proposal to advance to the next one."
        )
        st.json(current_key_proposal)
    elif loaded_key_proposals:
        st.info("No pending key proposals are loaded for review.")
        st.json(loaded_key_proposals)


with tabs[7]:
    st.subheader("Program Generator")
    program_user_id = _clean_id_input(st.text_input("Program user ID", key="program_user_id"))
    program_conversation_id = _clean_id_input(
        st.text_input("Program conversation ID", value="program_conversation", key="program_conversation_id")
    )
    if program_user_id:
        active_plan_response = httpx.get(
            f"{API_BASE_URL}/api/memory/plans/active",
            params={"user_id": program_user_id},
            timeout=30,
        )
        with st.expander("Active plan and progress"):
            if active_plan_response.is_success and active_plan_response.json():
                st.json(active_plan_response.json())
            elif active_plan_response.is_success:
                st.info("No active plan.")
            else:
                st.error(active_plan_response.json().get("detail", active_plan_response.text))
    if "program_messages" not in st.session_state:
        st.session_state["program_messages"] = []
    for item in st.session_state["program_messages"]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    program_message = st.chat_input("Ask for a new plan, revision, or rationale")
    if program_message:
        if not program_user_id:
            st.error("Program user ID is required.")
        else:
            st.session_state["program_messages"].append({"role": "user", "content": program_message})
            response = httpx.post(
                f"{API_BASE_URL}/api/programs/conversations/{program_conversation_id}/turns",
                json={"user_id": program_user_id, "message": program_message},
                timeout=300,
            )
            if response.is_success:
                payload = response.json()
                st.session_state["program_last_response"] = payload
                st.session_state["program_messages"].append({"role": "assistant", "content": payload["response"]})
                _rerun_app()
            else:
                st.error(response.json().get("detail", response.text))

    program_result = st.session_state.get("program_last_response")
    if program_result:
        draft_program_id = program_result.get("draft_program_id")
        draft_version_id = program_result.get("draft_version_id")
        review_reason = st.text_input("Review note", key="program_review_reason")
        requested_change = st.text_input("Requested plan change", key="program_requested_change")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Approve plan", disabled=not bool(draft_version_id)):
            response = httpx.post(
                f"{API_BASE_URL}/api/programs/{draft_program_id}/versions/{draft_version_id}/review",
                json={"user_id": program_user_id, "decision": "approve", "reason": review_reason or None},
                timeout=120,
            )
            st.session_state["program_review_result"] = response.json()
        if col2.button("Reject plan", disabled=not bool(draft_version_id)):
            response = httpx.post(
                f"{API_BASE_URL}/api/programs/{draft_program_id}/versions/{draft_version_id}/review",
                json={"user_id": program_user_id, "decision": "reject", "reason": review_reason or None},
                timeout=120,
            )
            st.session_state["program_review_result"] = response.json()
        if col3.button("Request change", disabled=not bool(draft_version_id and requested_change)):
            response = httpx.post(
                f"{API_BASE_URL}/api/programs/conversations/{program_conversation_id}/turns",
                json={"user_id": program_user_id, "message": requested_change},
                timeout=300,
            )
            if response.is_success:
                payload = response.json()
                st.session_state["program_last_response"] = payload
                st.session_state["program_messages"].extend(
                    [
                        {"role": "user", "content": requested_change},
                        {"role": "assistant", "content": payload["response"]},
                    ]
                )
                _rerun_app()
            else:
                st.error(response.json().get("detail", response.text))
        if col4.button("Refresh weekly notes", disabled=not bool(program_user_id)):
            response = httpx.post(
                f"{API_BASE_URL}/api/training/summary-notes/refresh",
                json={"user_id": program_user_id},
                timeout=120,
            )
            st.session_state["program_weekly_notes"] = response.json()
        if draft_program_id and draft_version_id:
            version = httpx.get(
                f"{API_BASE_URL}/api/programs/{draft_program_id}/versions/{draft_version_id}",
                timeout=60,
            )
            if version.is_success:
                st.json(version.json())
        context_id = program_result.get("context_id")
        if context_id:
            context_response = httpx.get(
                f"{API_BASE_URL}/api/programs/contexts/{context_id}",
                timeout=60,
            )
            if context_response.is_success:
                context_payload = context_response.json()
                with st.expander("SOP progress and context"):
                    st.json(
                        {
                            "status": context_payload.get("status"),
                            "active_sop": context_payload.get("active_sop"),
                            "current_sop_step": context_payload.get("current_sop_step"),
                            "pending_action": context_payload.get("pending_action"),
                            "sop_progress": context_payload.get("sections", {}).get("sop_progress", {}),
                            "evidence": context_payload.get("sections", {}).get("retrieved_evidence", []),
                            "calculations": context_payload.get("sections", {}).get("calculation_results", []),
                            "safety_and_warnings": context_payload.get("sections", {}).get(
                                "safety_flags_and_budget_warnings", []
                            ),
                        }
                    )
        trace_id = program_result.get("trace_id")
        if trace_id:
            trace_response = httpx.get(f"{API_BASE_URL}/api/programs/traces/{trace_id}", timeout=60)
            if trace_response.is_success:
                with st.expander("Program trace"):
                    st.json(trace_response.json())
    if st.session_state.get("program_review_result"):
        review_result = st.session_state["program_review_result"]
        st.json(review_result)
        xunji_confirmed = st.checkbox("Confirm Xunji write-back", key="program_xunji_confirmed")
        if st.button(
            "Write approved plan to Xunji",
            disabled=not bool(xunji_confirmed and review_result.get("review_status") == "approved"),
        ):
            response = httpx.post(
                f"{API_BASE_URL}/api/programs/{review_result['program_id']}/versions/{review_result['version_id']}/xunji-write",
                json={"user_id": program_user_id, "confirmed": True},
                timeout=120,
            )
            st.session_state["program_xunji_result"] = response.json()
    if st.session_state.get("program_xunji_result"):
        st.json(st.session_state["program_xunji_result"])
    if st.session_state.get("program_weekly_notes"):
        st.json(st.session_state["program_weekly_notes"])
