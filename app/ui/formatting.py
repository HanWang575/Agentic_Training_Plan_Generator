"""Formatting helpers for the local Streamlit UI."""

import re


SECTION_HEADINGS = {
    "Short answer": "Short Answer",
    "Practical guidance": "Practical Guidance",
    "Retrieved evidence summary": "Retrieved Evidence Summary",
    "Evidence used": "Evidence Used",
    "Safety note": "Safety Note",
    "Provider note": "Provider Note",
}


def answer_to_markdown(answer: str) -> str:
    """Convert the API answer string into readable Markdown.

    Parameters:
        answer: model answer text with shape `section_label:\ncontent`.
    Returns:
        str: Markdown where known section labels become headings.
    """

    cleaned = answer.strip()
    for raw_heading, display_heading in SECTION_HEADINGS.items():
        pattern = rf"(?m)^{re.escape(raw_heading)}:\s*$"
        cleaned = re.sub(pattern, f"### {display_heading}", cleaned)
    return cleaned


def citations_to_rows(citations: list[dict]) -> list[dict]:
    """Convert API citations into table rows for Streamlit.

    Parameters:
        citations: list with shape `[citation_count]` from `/api/qa/ask`.
    Returns:
        list[dict]: rows with source, chunk, title, and location columns.
    """

    rows = []
    for citation in citations:
        rows.append(
            {
                "source_id": citation.get("source_id", ""),
                "chunk_id": citation.get("chunk_id", ""),
                "title": citation.get("title", ""),
                "location": citation.get("page_or_section") or "",
            }
        )
    return rows
