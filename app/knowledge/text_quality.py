"""Text cleanup and quality checks for parsed source documents."""

import re


CJK_RE = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"


def clean_pdf_text(text: str) -> str:
    """Repair common PDF text extraction artifacts.

    Parameters:
        text: raw text from one PDF page or layout block.
    Returns:
        str: cleaned text suitable for chunking and embedding.
    """

    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00ad", "")
        .replace("\xa0", " ")
    )
    normalized = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", normalized)
    paragraphs = re.split(r"\n\s*\n+", normalized)
    cleaned_paragraphs = [_clean_paragraph(paragraph) for paragraph in paragraphs]
    cleaned = "\n\n".join(part for part in cleaned_paragraphs if part)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def assess_text_quality(text: str, label: str | None = None) -> list[str]:
    """Return warnings for text that still looks poorly extracted.

    Parameters:
        text: parsed text to inspect.
        label: optional page/chunk label included in each warning.
    Returns:
        list[str]: human-readable warnings with shape `[warning_count]`.
    """

    prefix = f"{label}: " if label else ""
    warnings: list[str] = []
    stripped = text.strip()
    if not stripped:
        return warnings

    nonblank_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(nonblank_lines) >= 6:
        average_line_length = sum(len(line) for line in nonblank_lines) / len(nonblank_lines)
        line_breaks_per_100_chars = max(0, len(nonblank_lines) - 1) / max(1, len(stripped)) * 100
        if average_line_length < 35 and line_breaks_per_100_chars > 3:
            warnings.append(
                f"{prefix}many short lines; PDF line wrapping may have damaged sentence flow"
            )

    tokens = re.findall(r"\S+", stripped)
    if len(tokens) >= 20:
        single_letter_tokens = [token for token in tokens if re.fullmatch(r"[A-Za-z]", token)]
        if len(single_letter_tokens) / len(tokens) > 0.35:
            warnings.append(
                f"{prefix}many isolated single-letter tokens; PDF text may be character-split"
            )

    if re.search(r"(?:\b[A-Za-z]\b\s*){8,}", stripped):
        warnings.append(f"{prefix}long run of spaced letters detected")

    return warnings


def _clean_paragraph(paragraph: str) -> str:
    """Clean and join lines inside one paragraph-like PDF text region."""

    lines = [_repair_spaced_characters(line.strip()) for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return ""

    output = lines[0]
    for next_line in lines[1:]:
        if _should_keep_line_break(output, next_line):
            output = f"{output}\n{next_line}"
        elif _touches_cjk(output[-1], next_line[0]):
            output = f"{output}{next_line}"
        else:
            output = f"{output} {next_line}"
    return output.strip()


def _repair_spaced_characters(line: str) -> str:
    """Repair words extracted as separated letters, such as `t r a i n i n g`."""

    parts = re.split(r"(\s{2,})", line)
    repaired_parts = []
    for part in parts:
        if not part.strip():
            repaired_parts.append(" ")
            continue
        tokens = part.split()
        if len(tokens) >= 3 and all(re.fullmatch(r"[A-Za-z]", token) for token in tokens):
            has_lowercase = any(token.islower() for token in tokens)
            if has_lowercase or len(tokens) >= 5:
                repaired_parts.append("".join(tokens))
                continue
        repaired_parts.append(part)
    return re.sub(r"[ \t]+", " ", "".join(repaired_parts)).strip()


def _should_keep_line_break(previous: str, current: str) -> bool:
    """Return true when a line break likely represents structure, not wrapping."""

    previous_line = previous.splitlines()[-1].strip()
    if _is_list_item(current):
        return True
    if _is_list_item(previous_line):
        return True
    if _looks_like_heading(previous_line) and not previous_line.endswith((".", ",", ";", ":")):
        return True
    return False


def _is_list_item(line: str) -> bool:
    """Detect bullet and numbered list starts."""

    return bool(re.match(r"^([-*•]|\(?\d+[.)]|\(?[A-Za-z][.)])\s+", line.strip()))


def _looks_like_heading(line: str) -> bool:
    """Detect short standalone lines that are likely headings."""

    clean = line.strip()
    if len(clean) > 80 or len(clean.split()) > 10:
        return False
    if clean.endswith((".", ",", ";")):
        return False
    if re.search(r"[:：]$", clean):
        return True
    letters = re.sub(r"[^A-Za-z]", "", clean)
    return bool(letters) and len(letters) >= 4 and letters.upper() == letters


def _touches_cjk(left: str, right: str) -> bool:
    """Return true when a wrapped line joins two Chinese/Japanese/Korean chars."""

    return bool(re.match(f"[{CJK_RE}]", left) and re.match(f"[{CJK_RE}]", right))
