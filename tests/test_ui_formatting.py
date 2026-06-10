"""Formatting tests for the Streamlit UI."""

from app.ui.formatting import answer_to_markdown, citations_to_rows


def test_answer_to_markdown_promotes_known_sections():
    """Plain answer labels should become readable Markdown headings."""

    answer = (
        "Short answer:\n"
        "Run mostly easy miles.\n\n"
        "Practical guidance:\n"
        "- Add threshold work gradually.\n\n"
        "Evidence used:\n"
        "- [src_a / chk_a] Source, page 1\n\n"
        "Safety note:\n"
        "Progress slowly."
    )

    markdown = answer_to_markdown(answer)

    assert "### Short Answer" in markdown
    assert "### Practical Guidance" in markdown
    assert "### Evidence Used" in markdown
    assert "### Safety Note" in markdown
    assert "Short answer:" not in markdown


def test_citations_to_rows_keeps_table_columns():
    """Citation dictionaries should become stable table rows."""

    rows = citations_to_rows(
        [
            {
                "source_id": "src_a",
                "chunk_id": "chk_a",
                "title": "Running Source",
                "page_or_section": "page 3",
            }
        ]
    )

    assert rows == [
        {
            "source_id": "src_a",
            "chunk_id": "chk_a",
            "title": "Running Source",
            "location": "page 3",
        }
    ]
