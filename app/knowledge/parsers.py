"""Document parsers that convert supported uploads into plain text blocks."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import UnsupportedFileTypeError
from app.knowledge.marker_parser import parse_pdf_with_marker
from app.knowledge.schemas import ParsedDocument, ParsedTextBlock, SUPPORTED_FILE_TYPES
from app.knowledge.text_quality import assess_text_quality, clean_pdf_text


def parse_document(
    file_path: Path,
    title: str | None = None,
    settings: Settings | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> ParsedDocument:
    """Parse one supported file into plain text blocks.

    Parameters:
        file_path: path to a `.pdf`, `.epub`, `.md`, `.txt`, or `.json` file.
        title: optional display title. Defaults to the file stem.
        settings: optional runtime settings. Needed for Marker PDF parsing.
        progress_callback: optional `(progress, message)` callback where
            `progress` is a float in `[0.0, 1.0]`.
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
    parser_warnings: list[str] = []
    if suffix == ".pdf":
        blocks, parser_warnings = _parse_pdf(
            file_path,
            settings=settings,
            progress_callback=progress_callback,
        )
    elif suffix == ".epub":
        blocks = _parse_epub(file_path)
    elif suffix == ".json":
        blocks = _parse_json(file_path)
    else:
        blocks = _parse_text(file_path)
    return ParsedDocument(
        title=display_title,
        source_type=suffix.lstrip("."),
        blocks=blocks,
        parser_warnings=parser_warnings,
    )


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


def _parse_pdf(
    file_path: Path,
    settings: Settings | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[ParsedTextBlock], list[str]]:
    """Extract PDF text with Marker when configured, otherwise PyMuPDF."""

    backend = (settings.pdf_parser_backend if settings else "pymupdf").strip().lower()
    if backend == "marker":
        try:
            return parse_pdf_with_marker(
                file_path,
                settings or Settings(_env_file=None),
                progress_callback=progress_callback,
            )
        except Exception as exc:
            blocks, warnings = _parse_pdf_with_pymupdf(file_path, progress_callback=progress_callback)
            return blocks, [f"Marker parser failed; fell back to PyMuPDF: {exc}", *warnings]
    if backend not in {"pymupdf", "fitz"}:
        blocks, warnings = _parse_pdf_with_pymupdf(file_path, progress_callback=progress_callback)
        return blocks, [f"Unknown PDF parser backend '{backend}'; used PyMuPDF.", *warnings]
    return _parse_pdf_with_pymupdf(file_path, progress_callback=progress_callback)


def _parse_pdf_with_pymupdf(
    file_path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[ParsedTextBlock], list[str]]:
    """Extract layout-aware text from PDF pages with PyMuPDF."""

    import fitz

    blocks: list[ParsedTextBlock] = []
    parser_warnings: list[str] = []
    with fitz.open(file_path) as document:
        total_pages = max(len(document), 1)
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            raw_text = _extract_pdf_page_text(page)
            raw_warnings = assess_text_quality(raw_text, label=f"page {page_number} raw")
            text = clean_pdf_text(raw_text)
            cleaned_warnings = assess_text_quality(text, label=f"page {page_number} cleaned")
            quality_warnings = cleaned_warnings or (
                [f"page {page_number}: repaired suspicious PDF extraction artifacts"]
                if raw_warnings
                else []
            )
            parser_warnings.extend(quality_warnings)
            if text:
                blocks.append(
                    ParsedTextBlock(
                        text=text,
                        page_number=page_number,
                        quality_warnings=quality_warnings,
                    )
                )
            if progress_callback is not None:
                progress_callback(page_number / total_pages, f"Parsing PDF page {page_number} of {total_pages}")
    return blocks, _dedupe_warnings(parser_warnings)


def _extract_pdf_page_text(page) -> str:
    """Read a PDF page using layout blocks, falling back to plain text."""

    text_blocks = []
    for block in page.get_text("blocks", sort=True):
        block_text = block[4].strip() if len(block) > 4 else ""
        block_type = block[6] if len(block) > 6 else 0
        if block_type != 0 or not block_text:
            continue
        text_blocks.append(clean_pdf_text(block_text))
    if text_blocks:
        return "\n\n".join(text_blocks)
    return page.get_text("text").strip()


def _dedupe_warnings(warnings: list[str], limit: int = 50) -> list[str]:
    """Keep parser warnings readable for source review screens."""

    unique_warnings = list(dict.fromkeys(warnings))
    if len(unique_warnings) <= limit:
        return unique_warnings
    hidden_count = len(unique_warnings) - limit
    return unique_warnings[:limit] + [f"{hidden_count} additional parser warnings hidden"]


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
