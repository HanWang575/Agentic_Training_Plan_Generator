"""Simple, readable document chunking utilities."""

import re

from app.knowledge.schemas import ChunkDraft, ParsedDocument, ParsedTextBlock


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
) -> list[ChunkDraft]:
    """Split parsed text into chunk drafts.

    Parameters:
        parsed_document: parser output with `[ParsedTextBlock]`.
        target_tokens: desired chunk size, default 700 from settings.
        overlap_tokens: repeated context between adjacent chunks, default 100.
    Returns:
        list[ChunkDraft]: each draft has text, index, page/section, and token count.
    """

    chunks: list[ChunkDraft] = []
    for block in parsed_document.blocks:
        chunks.extend(_chunk_block(block, target_tokens, overlap_tokens, len(chunks)))
    return chunks


def _chunk_block(
    block: ParsedTextBlock,
    target_tokens: int,
    overlap_tokens: int,
    start_index: int,
) -> list[ChunkDraft]:
    """Chunk one parsed block using paragraph boundaries first."""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", block.text) if part.strip()]
    windows: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        split_paragraphs = _split_large_paragraph(paragraph, target_tokens)
        for paragraph_part in split_paragraphs:
            current, current_tokens, windows = _add_paragraph_to_window(
                paragraph_part,
                current,
                current_tokens,
                windows,
                target_tokens,
                overlap_tokens,
            )

    if current:
        windows.append("\n\n".join(current))

    drafts: list[ChunkDraft] = []
    for offset, text in enumerate(windows):
        drafts.append(
            ChunkDraft(
                text=text,
                chunk_index=start_index + offset,
                page_number=block.page_number,
                section_heading=block.section_heading,
                token_count=estimate_token_count(text),
            )
        )
    return drafts


def _add_paragraph_to_window(
    paragraph: str,
    current: list[str],
    current_tokens: int,
    windows: list[str],
    target_tokens: int,
    overlap_tokens: int,
) -> tuple[list[str], int, list[str]]:
    """Add one paragraph to the active chunk window."""

    paragraph_tokens = estimate_token_count(paragraph)
    if current and current_tokens + paragraph_tokens > target_tokens:
        windows.append("\n\n".join(current))
        current = _overlap_tail(current, overlap_tokens)
        current_tokens = estimate_token_count("\n\n".join(current)) if current else 0
    current.append(paragraph)
    current_tokens += paragraph_tokens
    return current, current_tokens, windows


def _split_large_paragraph(paragraph: str, target_tokens: int) -> list[str]:
    """Fallback split for paragraphs larger than the target size."""

    if estimate_token_count(paragraph) <= target_tokens:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and estimate_token_count(" ".join(current + [sentence])) > target_tokens:
            parts.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        parts.append(" ".join(current))
    return parts


def _overlap_tail(paragraphs: list[str], overlap_tokens: int) -> list[str]:
    """Return enough trailing paragraphs to approximate the overlap size."""

    if overlap_tokens <= 0:
        return []
    selected: list[str] = []
    token_total = 0
    for paragraph in reversed(paragraphs):
        selected.insert(0, paragraph)
        token_total += estimate_token_count(paragraph)
        if token_total >= overlap_tokens:
            break
    return selected
