"""Minimal Streamlit UI for local review, chat, and trace inspection."""

import sys
import time
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


def _upload_rag_corpus_with_progress() -> dict:
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
                params={"relative_path": item["relative_path"]},
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


def _upload_source_with_progress(uploaded_file, title: str, topic_tags: str) -> dict:
    """Upload one source through the async endpoint and show parse progress."""

    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    data = {"title": title, "topic_tags": topic_tags}
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


st.set_page_config(page_title="Evidence Q&A", layout="wide")
st.title("Evidence-Based Q&A Assistant")

tabs = st.tabs(["Chat", "Upload", "Review", "Sources", "Trace"])


with tabs[0]:
    st.subheader("Ask")
    question = st.text_area("Question", height=120)
    topic_hint = st.text_input("Topic hint")
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
            json={"question": question, "topic_hint": topic_hint or None, "debug": True},
            timeout=120,
        )
        response.raise_for_status()
        _render_qa_response(response.json(), show_debug=debug)


with tabs[1]:
    st.subheader("Upload Source")
    uploaded_file = st.file_uploader("Source file", type=["pdf", "epub", "md", "txt", "json"])
    upload_title = st.text_input("Title", placeholder="Auto from file name")
    upload_tags = st.text_input("Topic tags, comma separated", placeholder="Optional")
    if st.button("Upload") and uploaded_file is not None:
        st.json(_upload_source_with_progress(uploaded_file, upload_title, upload_tags))
    if st.button("Upload RAG_corpus"):
        st.json(_upload_rag_corpus_with_progress())


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
    trace_id = st.text_input("Trace ID")
    if st.button("Load trace") and trace_id:
        response = httpx.get(f"{API_BASE_URL}/api/qa/traces/{trace_id}", timeout=30)
        response.raise_for_status()
        _render_trace_response(response.json())
