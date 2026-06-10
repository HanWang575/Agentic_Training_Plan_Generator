"""Minimal Streamlit UI for local review, chat, and trace inspection."""

import sys
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


def _render_qa_response(payload: dict) -> None:
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

    if payload.get("debug"):
        with st.expander("Debug trace"):
            st.json(payload["debug"])
    with st.expander("Raw API response"):
        st.json(payload)


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
            json={"question": question, "topic_hint": topic_hint or None, "debug": debug},
            timeout=120,
        )
        response.raise_for_status()
        _render_qa_response(response.json())


with tabs[1]:
    st.subheader("Upload Source")
    uploaded_file = st.file_uploader("Source file", type=["pdf", "epub", "md", "txt", "json"])
    upload_title = st.text_input("Title", placeholder="Auto from file name")
    upload_tags = st.text_input("Topic tags, comma separated", placeholder="Optional")
    if st.button("Upload") and uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {"title": upload_title, "topic_tags": upload_tags}
        response = httpx.post(f"{API_BASE_URL}/api/knowledge/upload", files=files, data=data, timeout=120)
        st.json(response.json())
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
        st.json(response.json())
