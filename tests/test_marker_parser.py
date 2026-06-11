"""Marker parsing and hybrid chunking behavior."""

import subprocess

from app.core.config import Settings
from app.knowledge.chunking import chunk_document
from app.knowledge.marker_parser import (
    _display_page_range,
    _marker_page_ranges,
    markdown_to_blocks,
    parse_pdf_with_marker,
)
from app.knowledge.schemas import ParsedDocument, ParsedTextBlock


def test_marker_markdown_table_keeps_footnote_without_guessed_title_or_description():
    """Table parsing should keep structural metadata but not guess captions."""

    markdown = """# Four-Day Template

This table describes a four-day strength plan.

| Monday | Tuesday | Thursday | Friday |
| --- | --- | --- | --- |
| Squat 5/3/1 | Bench 5/3/1 | Deadlift 5/3/1 | Press 5/3/1 |
| Assistance | Assistance | Assistance | Assistance |
Note: Use jumps and throws before lifting.
"""

    blocks = markdown_to_blocks(markdown)
    table = next(block for block in blocks if block.content_type == "table")

    assert table.section_heading == "Four-Day Template"
    assert "table_title" not in table.content_metadata
    assert "table_description" not in table.content_metadata
    assert table.content_metadata["table_footnote"] == "Note: Use jumps and throws before lifting."
    assert table.content_metadata["table_format"] == "markdown"


def test_marker_markdown_image_keeps_alt_text_and_path():
    """Marker image references should preserve concrete image metadata only."""

    markdown = """# Jump Training

The following image shows a box jump sequence.

![Box jump sequence](images/box_jump.png)
"""

    blocks = markdown_to_blocks(markdown)
    figure = next(block for block in blocks if block.content_type == "figure")

    assert figure.content_metadata["image_alt_text"] == "Box jump sequence"
    assert figure.content_metadata["image_path"] == "images/box_jump.png"


def test_chunk_document_adds_neighbor_paragraphs_to_small_table_chunks():
    """Small tables should carry nearby prose context into the final chunk text."""

    markdown = """# Plan

This is the first setup paragraph.

This is the second setup paragraph.

| Day | Lift |
| --- | --- |
| 1 | Squat |

This is the next paragraph after the table.
"""

    parsed_document = ParsedDocument(
        title="Plan",
        source_type="md",
        blocks=markdown_to_blocks(markdown),
    )

    chunks = chunk_document(
        parsed_document,
        target_tokens=512,
        overlap_tokens=128,
        maximum_table_chunk_size=2048,
    )
    table_chunk = next(chunk for chunk in chunks if chunk.content_type == "table")

    assert "Previous paragraphs:" in table_chunk.text
    assert "This is the first setup paragraph." in table_chunk.text
    assert "This is the second setup paragraph." in table_chunk.text
    assert "Next paragraph:" in table_chunk.text
    assert "This is the next paragraph after the table." in table_chunk.text
    assert table_chunk.content_metadata["previous_paragraphs"] == [
        "This is the first setup paragraph.",
        "This is the second setup paragraph.",
    ]
    assert table_chunk.content_metadata["next_paragraphs"] == [
        "This is the next paragraph after the table."
    ]


def test_chunk_document_adds_neighbor_paragraphs_to_small_image_chunks():
    """Small figures should also carry nearby prose context into the chunk text."""

    markdown = """# Jumps

This is the first setup paragraph.

This is the second setup paragraph.

![Box jump](images/box_jump.png)

This is the next paragraph after the image.
"""

    parsed_document = ParsedDocument(
        title="Jumps",
        source_type="md",
        blocks=markdown_to_blocks(markdown),
    )

    chunks = chunk_document(
        parsed_document,
        target_tokens=512,
        overlap_tokens=128,
        maximum_table_chunk_size=2048,
    )
    figure_chunk = next(chunk for chunk in chunks if chunk.content_type == "figure")

    assert "Previous paragraphs:" in figure_chunk.text
    assert "This is the first setup paragraph." in figure_chunk.text
    assert "This is the second setup paragraph." in figure_chunk.text
    assert "Next paragraph:" in figure_chunk.text
    assert "This is the next paragraph after the image." in figure_chunk.text
    assert figure_chunk.content_metadata["previous_paragraphs"] == [
        "This is the first setup paragraph.",
        "This is the second setup paragraph.",
    ]
    assert figure_chunk.content_metadata["next_paragraphs"] == [
        "This is the next paragraph after the image."
    ]


def test_chunk_document_uses_recursive_splitter_for_large_plain_text_sections():
    """Large prose sections should be recursively split under the token budget."""

    repeated_paragraph = "Strength training improves force production and skill when practice stays consistent."
    large_text = "\n\n".join(repeated_paragraph for _ in range(8))
    parsed_document = ParsedDocument(
        title="Large Section",
        source_type="md",
        blocks=[
            ParsedTextBlock(
                text="Methods",
                section_heading="Methods",
                content_type="plain_text",
                content_metadata={"role": "heading", "heading_level": 1},
            ),
            ParsedTextBlock(
                text=large_text,
                section_heading="Methods",
                content_type="plain_text",
            ),
        ],
    )

    chunks = chunk_document(
        parsed_document,
        target_tokens=40,
        overlap_tokens=10,
    )

    plain_chunks = [chunk for chunk in chunks if chunk.content_type == "plain_text"]
    assert len(plain_chunks) > 1
    assert all(chunk.section_heading == "Methods" for chunk in plain_chunks)
    assert all(chunk.content_metadata["chunking_strategy"] == "markdown_header_recursive" for chunk in plain_chunks)


def test_large_table_is_split_and_marked_as_part_of_larger_table():
    """Oversized tables should be split and labeled for downstream reasoning."""

    large_table_text = "Content type: table\n\n" + "\n".join(
        f"| Day {index} | Main lift {index} | Assistance {index} |"
        for index in range(40)
    )
    parsed_document = ParsedDocument(
        title="Large Table",
        source_type="pdf",
        blocks=[
            ParsedTextBlock(
                text=large_table_text,
                section_heading="Programming",
                content_type="table",
                content_metadata={"table_format": "markdown"},
            )
        ],
    )

    chunks = chunk_document(
        parsed_document,
        target_tokens=512,
        overlap_tokens=128,
        maximum_table_chunk_size=60,
    )

    assert len(chunks) > 1
    assert all(chunk.content_type == "table" for chunk in chunks)
    assert all(chunk.content_metadata["structured_part_count"] == len(chunks) for chunk in chunks)
    assert all(chunk.content_metadata["is_partial_larger_table"] is True for chunk in chunks)
    assert "part 1 of" in chunks[0].text


def test_table_note_with_fullwidth_colon_is_captured():
    """Marker table footnotes may use a fullwidth colon in non-English PDFs."""

    markdown = """| Day | Lift |
| --- | --- |
| 1 | Squat |
Note\uFF1AKeep one rep in reserve.
"""

    blocks = markdown_to_blocks(markdown)
    table = next(block for block in blocks if block.content_type == "table")

    assert table.content_metadata["table_footnote"] == "Note：Keep one rep in reserve."


def test_parse_pdf_with_marker_uses_absolute_pdf_path(monkeypatch, tmp_path):
    """Marker should receive an absolute file path when upload storage uses relative paths."""

    relative_pdf = Settings(_env_file=None).upload_dir / "src_test" / "sample.pdf"
    expected_pdf = relative_pdf.resolve()
    captured: dict[str, object] = {}

    def fake_run(command, cwd, env, capture_output, text, encoding, errors, timeout):
        captured["command"] = command
        captured["cwd"] = cwd
        markdown_path = tmp_path / "marker_output" / "sample" / "sample.md"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text("| Day | Lift |\n| --- | --- |\n| 1 | Squat |\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_find_marker_markdown(_output_dir):
        return tmp_path / "marker_output" / "sample" / "sample.md"

    monkeypatch.setattr("app.knowledge.marker_parser._marker_single_from_venv", lambda: "marker_single")
    monkeypatch.setattr("app.knowledge.marker_parser._marker_page_ranges", lambda _path, _size: ["0-1"])
    monkeypatch.setattr("app.knowledge.marker_parser.subprocess.run", fake_run)
    monkeypatch.setattr("app.knowledge.marker_parser._find_marker_markdown", fake_find_marker_markdown)

    blocks, warnings = parse_pdf_with_marker(
        relative_pdf,
        Settings(_env_file=None, marker_torch_device="cuda", marker_force_ocr=False),
    )

    assert blocks
    assert warnings
    assert captured["command"][1] == str(expected_pdf)
    assert captured["cwd"] == str(expected_pdf.parent)


def test_marker_page_ranges_are_zero_based(tmp_path):
    """Marker expects zero-based inclusive page ranges."""

    import fitz

    pdf_path = tmp_path / "pages.pdf"
    document = fitz.open()
    for _ in range(12):
        document.new_page()
    document.save(pdf_path)
    document.close()

    assert _marker_page_ranges(pdf_path, batch_size=5) == ["0-4", "5-9", "10-11"]
    assert _display_page_range("0-4") == "1-5"
    assert _display_page_range("10-11") == "11-12"
