"""PDF text cleanup and quality warning behavior."""

from app.knowledge.chunking import chunk_document
from app.knowledge.schemas import ParsedDocument, ParsedTextBlock
from app.knowledge.text_quality import assess_text_quality, clean_pdf_text


def test_clean_pdf_text_joins_soft_line_breaks_and_keeps_lists():
    """Wrapped prose is joined, but bullet structure remains readable."""

    raw_text = (
        "Endurance training is performed at low intensity\n"
        "for a long duration and improves aerobic capacity.\n\n"
        "- Easy running\n"
        "- Long cycling\n"
    )

    cleaned = clean_pdf_text(raw_text)

    assert "low intensity for a long duration" in cleaned
    assert "- Easy running\n- Long cycling" in cleaned


def test_clean_pdf_text_repairs_hyphenation_and_chinese_wraps():
    """English hyphenation and Chinese line wraps are repaired differently."""

    raw_text = "Resis-\ntance training builds strength.\n\n耐力训\n练可以提高有氧能力。"

    cleaned = clean_pdf_text(raw_text)

    assert "Resistance training" in cleaned
    assert "耐力训练可以提高有氧能力。" in cleaned


def test_clean_pdf_text_repairs_spaced_letters_when_likely_word_artifact():
    """Single-letter word artifacts are joined when they look like one word."""

    cleaned = clean_pdf_text("T h e\ns t r e n g t h\nphase")

    assert "The strength phase" in cleaned


def test_assess_text_quality_flags_character_split_text():
    """Quality checks identify text that still looks character-split."""

    warnings = assess_text_quality(
        "T h e p r o g r a m u s e s e n d u r a n c e a n d r e s i s t a n c e blocks.",
        label="chunk 1",
    )

    assert any("spaced letters" in warning or "single-letter" in warning for warning in warnings)


def test_chunk_document_carries_block_quality_warnings():
    """Parser warnings on a block are preserved on generated chunks."""

    parsed = ParsedDocument(
        title="Warning Source",
        source_type="pdf",
        blocks=[
            ParsedTextBlock(
                text="Clean enough text for one chunk.",
                page_number=1,
                quality_warnings=["page 1: repaired suspicious PDF extraction artifacts"],
            )
        ],
    )

    chunks = chunk_document(parsed, target_tokens=100, overlap_tokens=10)

    assert chunks[0].quality_warnings == ["page 1: repaired suspicious PDF extraction artifacts"]
