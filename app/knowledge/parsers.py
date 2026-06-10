"""Document parsers that convert supported uploads into plain text blocks."""

import json
from pathlib import Path
from typing import Any

from app.core.errors import UnsupportedFileTypeError
from app.knowledge.schemas import ParsedDocument, ParsedTextBlock, SUPPORTED_FILE_TYPES


def parse_document(file_path: Path, title: str | None = None) -> ParsedDocument:
    """Parse one supported file into plain text blocks.

    Parameters:
        file_path: path to a `.pdf`, `.epub`, `.md`, `.txt`, or `.json` file.
        title: optional display title. Defaults to the file stem.
    Returns:
        ParsedDocument: title, file type, and `[ParsedTextBlock]`.
    Raises:
        UnsupportedFileTypeError: if the extension is not supported.
    """

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_FILE_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported types: {sorted(SUPPORTED_FILE_TYPES)}"
        )

    display_title = title or file_path.stem
    if suffix == ".pdf":
        blocks = _parse_pdf(file_path)
    elif suffix == ".epub":
        blocks = _parse_epub(file_path)
    elif suffix == ".json":
        blocks = _parse_json(file_path)
    else:
        blocks = _parse_text(file_path)
    return ParsedDocument(title=display_title, source_type=suffix.lstrip("."), blocks=blocks)


def _parse_text(file_path: Path) -> list[ParsedTextBlock]:
    """Read Markdown or plain text as one block."""

    text = file_path.read_text(encoding="utf-8", errors="replace")
    return [ParsedTextBlock(text=text)]


def _parse_json(file_path: Path) -> list[ParsedTextBlock]:
    """Read JSON and emit logical records when possible.

    Output shape:
        list with one block per top-level object/list item, or one fallback block.
    """

    data = json.loads(file_path.read_text(encoding="utf-8"))
    records: list[Any] = data if isinstance(data, list) else [data]
    blocks = []
    for index, record in enumerate(records):
        text = json.dumps(record, ensure_ascii=False, indent=2)
        blocks.append(ParsedTextBlock(text=text, section_heading=f"JSON record {index + 1}"))
    return blocks


def _parse_pdf(file_path: Path) -> list[ParsedTextBlock]:
    """Extract text from PDF pages with PyMuPDF."""

    import fitz

    blocks: list[ParsedTextBlock] = []
    with fitz.open(file_path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                blocks.append(ParsedTextBlock(text=text, page_number=page_index + 1))
    return blocks


def _parse_epub(file_path: Path) -> list[ParsedTextBlock]:
    """Extract readable text from EPUB HTML items."""

    from bs4 import BeautifulSoup
    from ebooklib import ITEM_DOCUMENT, epub

    book = epub.read_epub(str(file_path))
    blocks: list[ParsedTextBlock] = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        heading = soup.find(["h1", "h2", "h3"])
        section_heading = heading.get_text(" ", strip=True) if heading else None
        text = soup.get_text("\n", strip=True)
        if text:
            blocks.append(ParsedTextBlock(text=text, section_heading=section_heading))
    return blocks
