"""Structure-aware document chunking built on LangChain text splitters."""

from dataclasses import dataclass
import re
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.knowledge.schemas import ChunkDraft, ParsedDocument, ParsedTextBlock
from app.knowledge.text_quality import assess_text_quality


HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ],
    strip_headers=False,
)


@dataclass(slots=True)
class _PlainSection:
    """One contiguous prose run under the same heading context."""

    heading: str | None
    heading_level: int
    page_number: int | None
    blocks: list[ParsedTextBlock]


def estimate_token_count(text: str) -> int:
    """Estimate token count from text.

    Parameters:
        text: any plain text string.
    Returns:
        int: approximate token count. This is intentionally simple for v1.
    """

    word_count = len(re.findall(r"\S+", text))
    return max(1, int(word_count * 1.3))


def chunk_document(
    parsed_document: ParsedDocument,
    target_tokens: int,
    overlap_tokens: int,
    maximum_table_chunk_size: int = 2048,
    structured_context_previous_paragraphs: int = 2,
    structured_context_next_paragraphs: int = 1,
) -> list[ChunkDraft]:
    """Split parsed text with section-first prose chunking and structured rules.

    Parameters:
        parsed_document: parser output with `[ParsedTextBlock]`.
        target_tokens: desired prose chunk size, default 512 from settings.
        overlap_tokens: repeated prose context between adjacent chunks, default 128.
        maximum_table_chunk_size: max token size for atomic tables/images.
        structured_context_previous_paragraphs: number of previous prose paragraphs
            to attach to small table/image chunks.
        structured_context_next_paragraphs: number of later prose paragraphs to
            attach to small table/image chunks.
    Returns:
        list[ChunkDraft]: chunks with text, chunk index, metadata, and token count.
    """

    chunks: list[ChunkDraft] = []
    current_heading: str | None = None
    current_heading_level = 1
    current_plain_blocks: list[ParsedTextBlock] = []

    for block_index, block in enumerate(parsed_document.blocks):
        if _is_heading_block(block):
            chunks.extend(
                _flush_plain_section(
                    current_plain_blocks,
                    current_heading,
                    current_heading_level,
                    len(chunks),
                    target_tokens,
                    overlap_tokens,
                )
            )
            current_plain_blocks = []
            current_heading = block.text.strip() or current_heading
            current_heading_level = int(block.content_metadata.get("heading_level", 1))
            continue

        if block.content_type == "plain_text":
            current_plain_blocks.append(block)
            continue

        chunks.extend(
            _flush_plain_section(
                current_plain_blocks,
                current_heading,
                current_heading_level,
                len(chunks),
                target_tokens,
                overlap_tokens,
            )
        )
        current_plain_blocks = []
        chunks.extend(
            _structured_chunks(
                parsed_document.blocks,
                block_index,
                block,
                len(chunks),
                maximum_table_chunk_size,
                structured_context_previous_paragraphs,
                structured_context_next_paragraphs,
            )
        )

    chunks.extend(
        _flush_plain_section(
            current_plain_blocks,
            current_heading,
            current_heading_level,
            len(chunks),
            target_tokens,
            overlap_tokens,
        )
    )
    return chunks


def _flush_plain_section(
    blocks: list[ParsedTextBlock],
    heading: str | None,
    heading_level: int,
    start_index: int,
    target_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    """Chunk one contiguous prose run with header-first then recursive splitting."""

    if not blocks:
        return []
    section = _PlainSection(
        heading=heading,
        heading_level=max(1, heading_level),
        page_number=_first_page_number(blocks),
        blocks=blocks,
    )
    section_markdown = _section_markdown(section)
    if not section_markdown.strip():
        return []

    section_documents = _split_markdown_sections(section_markdown)
    drafts: list[ChunkDraft] = []
    quality_warnings = _dedupe_warnings(
        warning
        for block in blocks
        for warning in block.quality_warnings
    )
    for doc_offset, (text, metadata) in enumerate(section_documents):
        section_heading = _section_heading_from_metadata(metadata, heading)
        prose_parts = _split_prose_text(text, target_tokens, overlap_tokens)
        for part_offset, part_text in enumerate(prose_parts):
            chunk_index = start_index + len(drafts)
            part_warnings = [*quality_warnings]
            part_warnings.extend(assess_text_quality(part_text, label=f"chunk {chunk_index}"))
            content_metadata = {
                "chunking_strategy": "markdown_header_recursive",
                "section_headers": metadata,
            }
            if len(prose_parts) > 1:
                content_metadata["section_part_index"] = part_offset + 1
                content_metadata["section_part_count"] = len(prose_parts)
            drafts.append(
                ChunkDraft(
                    text=part_text,
                    chunk_index=chunk_index,
                    page_number=section.page_number,
                    section_heading=section_heading,
                    token_count=estimate_token_count(part_text),
                    content_type="plain_text",
                    content_metadata=content_metadata,
                    quality_warnings=_dedupe_warnings(part_warnings),
                )
            )
    return drafts


def _split_markdown_sections(markdown_text: str) -> list[tuple[str, dict[str, str]]]:
    """Use LangChain's Markdown header splitter on one section markdown string."""

    documents = HEADER_SPLITTER.split_text(markdown_text)
    if not documents:
        return [(markdown_text, {})]
    return [(document.page_content.strip(), dict(document.metadata)) for document in documents if document.page_content.strip()]


def _split_prose_text(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Return one prose chunk when small, else recursive LangChain chunks."""

    cleaned = text.strip()
    if not cleaned:
        return []
    if estimate_token_count(cleaned) <= target_tokens:
        return [cleaned]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        length_function=estimate_token_count,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            ". ",
            "! ",
            "? ",
            "; ",
            "；",
            ", ",
            "，",
            " ",
            "",
        ],
        keep_separator=True,
    )
    return [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]


def _structured_chunks(
    blocks: list[ParsedTextBlock],
    block_index: int,
    block: ParsedTextBlock,
    start_index: int,
    maximum_table_chunk_size: int,
    previous_paragraph_count: int,
    next_paragraph_count: int,
) -> list[ChunkDraft]:
    """Chunk one table or figure, optionally attaching local prose context."""

    token_count = estimate_token_count(block.text)
    if token_count <= maximum_table_chunk_size:
        previous_paragraphs = _neighbor_paragraphs(blocks, block_index, direction=-1, limit=previous_paragraph_count)
        next_paragraphs = _neighbor_paragraphs(blocks, block_index, direction=1, limit=next_paragraph_count)
        chunk_text = _contextual_structured_text(
            block,
            previous_paragraphs=previous_paragraphs,
            next_paragraphs=next_paragraphs,
        )
        quality_warnings = [*block.quality_warnings]
        quality_warnings.extend(assess_text_quality(chunk_text, label=f"chunk {start_index}"))
        return [
            ChunkDraft(
                text=chunk_text,
                chunk_index=start_index,
                page_number=block.page_number,
                section_heading=block.section_heading,
                token_count=estimate_token_count(chunk_text),
                content_type=block.content_type,
                content_metadata={
                    **block.content_metadata,
                    "chunking_strategy": "structured_atomic_contextual",
                    "previous_paragraphs": previous_paragraphs,
                    "next_paragraphs": next_paragraphs,
                },
                quality_warnings=_dedupe_warnings(quality_warnings),
            )
        ]

    parts = _split_large_structured_text(block.text, maximum_table_chunk_size)
    drafts: list[ChunkDraft] = []
    for offset, part_text in enumerate(parts):
        chunk_index = start_index + offset
        chunk_text = _structured_part_text(block, part_text, offset + 1, len(parts))
        quality_warnings = [*block.quality_warnings]
        quality_warnings.extend(assess_text_quality(chunk_text, label=f"chunk {chunk_index}"))
        content_metadata: dict[str, Any] = {
            **block.content_metadata,
            "chunking_strategy": "structured_recursive_split",
            "parent_content_type": block.content_type,
            "structured_part_index": offset + 1,
            "structured_part_count": len(parts),
        }
        if block.content_type == "table":
            content_metadata["is_partial_larger_table"] = True
        drafts.append(
            ChunkDraft(
                text=chunk_text,
                chunk_index=chunk_index,
                page_number=block.page_number,
                section_heading=block.section_heading,
                token_count=estimate_token_count(chunk_text),
                content_type=block.content_type,
                content_metadata=content_metadata,
                quality_warnings=_dedupe_warnings(quality_warnings),
            )
        )
    return drafts


def _split_large_structured_text(text: str, maximum_table_chunk_size: int) -> list[str]:
    """Split oversized table or figure text with recursive LangChain splitting."""

    overlap_tokens = min(128, max(0, maximum_table_chunk_size - 1))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=maximum_table_chunk_size,
        chunk_overlap=overlap_tokens,
        length_function=estimate_token_count,
        separators=[
            "\n|",
            "\n\n",
            "\n",
            " | ",
            " ",
            "",
        ],
        keep_separator=True,
    )
    return [chunk.strip() for chunk in splitter.split_text(text.strip()) if chunk.strip()]


def _contextual_structured_text(
    block: ParsedTextBlock,
    previous_paragraphs: list[str],
    next_paragraphs: list[str],
) -> str:
    """Attach nearby prose context to a small table or figure chunk."""

    parts: list[str] = []
    if block.section_heading:
        parts.append(f"Section heading: {block.section_heading}")
    if previous_paragraphs:
        parts.append("Previous paragraphs:\n" + "\n\n".join(previous_paragraphs))
    parts.append(block.text.strip())
    if next_paragraphs:
        label = "Next paragraph" if len(next_paragraphs) == 1 else "Next paragraphs"
        parts.append(f"{label}:\n" + "\n\n".join(next_paragraphs))
    return "\n\n".join(part for part in parts if part.strip())


def _structured_part_text(
    block: ParsedTextBlock,
    part_text: str,
    part_index: int,
    part_count: int,
) -> str:
    """Label one part of a larger structured chunk so the LLM sees the split."""

    parts: list[str] = []
    if block.section_heading:
        parts.append(f"Section heading: {block.section_heading}")
    parts.append(
        f"Structured chunk note: this is part {part_index} of {part_count} from a larger {block.content_type}."
    )
    parts.append(part_text.strip())
    return "\n\n".join(part for part in parts if part.strip())


def _neighbor_paragraphs(
    blocks: list[ParsedTextBlock],
    block_index: int,
    direction: int,
    limit: int,
) -> list[str]:
    """Collect nearby plain-text paragraphs around one structured block."""

    if limit <= 0:
        return []
    paragraphs: list[str] = []
    indexes = range(block_index - 1, -1, -1) if direction < 0 else range(block_index + 1, len(blocks))
    for index in indexes:
        block = blocks[index]
        if _is_heading_block(block):
            break
        if block.content_type != "plain_text":
            continue
        block_paragraphs = _paragraphs_from_text(block.text)
        if direction < 0:
            for paragraph in reversed(block_paragraphs):
                paragraphs.insert(0, paragraph)
                if len(paragraphs) >= limit:
                    return paragraphs[-limit:]
        else:
            for paragraph in block_paragraphs:
                paragraphs.append(paragraph)
                if len(paragraphs) >= limit:
                    return paragraphs[:limit]
    return paragraphs[-limit:] if direction < 0 else paragraphs[:limit]


def _section_markdown(section: _PlainSection) -> str:
    """Render one prose run back to markdown for header-aware splitting."""

    body = "\n\n".join(block.text.strip() for block in section.blocks if block.text.strip())
    if not body:
        return ""
    if not section.heading:
        return body
    prefix = "#" * max(1, section.heading_level)
    return f"{prefix} {section.heading}\n\n{body}"


def _section_heading_from_metadata(metadata: dict[str, str], fallback_heading: str | None) -> str | None:
    """Pick the deepest markdown header recorded by LangChain."""

    for key in ("h4", "h3", "h2", "h1"):
        value = metadata.get(key)
        if value:
            return value
    return fallback_heading


def _first_page_number(blocks: list[ParsedTextBlock]) -> int | None:
    """Return the first available page number from one block run."""

    for block in blocks:
        if block.page_number is not None:
            return block.page_number
    return None


def _is_heading_block(block: ParsedTextBlock) -> bool:
    """Return true when a parsed plain-text block is just a section heading."""

    return (
        block.content_type == "plain_text"
        and block.content_metadata.get("role") == "heading"
    )


def _paragraphs_from_text(text: str) -> list[str]:
    """Split prose into paragraph strings."""

    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _dedupe_warnings(warnings) -> list[str]:
    """Keep warning lists stable and duplicate-free."""

    return list(dict.fromkeys(warning for warning in warnings if warning))
