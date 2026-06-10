"""Q&A workflow behavior."""

from pathlib import Path

from sqlmodel import Session

from app.core.config import Settings
from app.knowledge.service import KnowledgeService
from app.qa.nodes import AnswerGenerator, CitationChecker
from app.knowledge.schemas import RetrievedChunk
from app.qa.schemas import CitationOut
from app.qa.service import QAService
from app.storage.database import create_app_engine, init_db


def _make_services(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        sqlite_path=tmp_path / "app.sqlite",
        upload_dir=tmp_path / "uploads",
        chroma_path=tmp_path / "chroma",
        openai_api_key=None,
    )
    engine = create_app_engine(settings)
    init_db(engine)
    session = Session(engine)
    knowledge = KnowledgeService(session=session, settings=settings)
    qa = QAService(session=session, settings=settings, knowledge_service=knowledge)
    return session, knowledge, qa


def test_unknown_question_returns_insufficient_evidence_and_trace(tmp_path: Path):
    """No approved chunks means the assistant refuses to guess and stores a trace."""

    session, _, qa = _make_services(tmp_path)

    response = qa.ask("What is the best marathon taper?", topic_hint="running", debug=True)
    trace = qa.get_trace(response.trace_id)

    assert response.insufficient_evidence is True
    assert response.citations == []
    assert "not have enough approved evidence" in response.answer
    assert trace.trace_id == response.trace_id
    assert trace.final_chunk_ids == []
    session.close()


def test_safety_question_adds_safety_note(tmp_path: Path):
    """Injury, medical, doping, and extreme diet terms are visible in safety output."""

    session, _, qa = _make_services(tmp_path)

    response = qa.ask("Can I train through knee injury pain?", debug=False)

    assert response.safety_note is not None
    assert "injury" in response.safety_note
    session.close()


def test_citation_checker_scrubs_unsupported_citation():
    """Citation checking removes markers that were not in retrieved context."""

    checker = CitationChecker()
    result = checker(
        {
            "draft_answer": "Evidence used:\n- [src_fake / chk_fake] invented",
            "citations": [
                CitationOut(
                    source_id="src_real",
                    chunk_id="chk_real",
                    title="Real Source",
                    page_or_section="page 1",
                )
            ],
            "errors": [],
            "timings": {},
        }
    )

    assert "[src_fake / chk_fake]" not in result["final_answer"]
    assert "[unsupported citation removed]" in result["final_answer"]


def test_citation_checker_returns_only_citations_used_in_answer():
    """The API citation list should match what the final answer actually cited."""

    checker = CitationChecker()
    result = checker(
        {
            "draft_answer": (
                "Short answer:\n"
                "Insufficient evidence to determine a daily protein target.\n\n"
                "Evidence used:\n"
                "- [src_a / chk_a] Protein Note, section unavailable"
            ),
            "citations": [
                CitationOut(
                    source_id="src_a",
                    chunk_id="chk_a",
                    title="Protein Note",
                    page_or_section="section unavailable",
                ),
                CitationOut(
                    source_id="src_b",
                    chunk_id="chk_b",
                    title="Nutrition PDF",
                    page_or_section="page 1",
                ),
                CitationOut(
                    source_id="src_b",
                    chunk_id="chk_c",
                    title="Nutrition PDF",
                    page_or_section="page 3",
                ),
            ],
            "insufficient_evidence": False,
            "errors": [],
            "timings": {},
        }
    )

    assert result["insufficient_evidence"] is True
    assert result["citations"] == [
        CitationOut(
            source_id="src_a",
            chunk_id="chk_a",
            title="Protein Note",
            page_or_section="section unavailable",
        )
    ]


def test_semantic_provider_failure_returns_insufficient_evidence_not_crash(tmp_path: Path):
    """Provider/network errors during semantic retrieval become trace errors."""

    class FailingProvider:
        def embed_texts(self, texts):
            raise RuntimeError("simulated embedding timeout")

    settings = Settings(
        _env_file=None,
        sqlite_path=tmp_path / "app.sqlite",
        upload_dir=tmp_path / "uploads",
        chroma_path=tmp_path / "chroma",
        openai_api_key="dummy-key",
    )
    engine = create_app_engine(settings)
    init_db(engine)
    session = Session(engine)
    qa = QAService(session=session, settings=settings, llm_provider=FailingProvider())

    response = qa.ask("How much protein should I eat to build muscle?", topic_hint="nutrition", debug=True)

    assert response.insufficient_evidence is True
    assert "Semantic retrieval failed" in response.debug["errors"][0]
    session.close()


def test_answer_generation_failure_returns_evidence_fallback():
    """If chat generation fails after retrieval, keep citations and show evidence snippets."""

    class FailingAnswerProvider:
        def generate_answer(self, request):
            raise RuntimeError("simulated chat quota error")

    node = AnswerGenerator(
        Settings(_env_file=None),
        FailingAnswerProvider(),
    )
    state = {
        "clean_question": "How much protein should I eat to build muscle?",
        "evidence_context": "[src_test / chk_test] Protein Note\nProtein supports muscle repair.",
        "context_results": [
            RetrievedChunk(
                source_id="src_test",
                chunk_id="chk_test",
                title="Protein Note",
                text="Protein supports muscle repair.",
                page_number=None,
                section_heading=None,
                topic_tags=["nutrition"],
                token_count=5,
                score=1.0,
                rank=1,
                retrieval_method="keyword",
            )
        ],
        "citations": [
            CitationOut(
                source_id="src_test",
                chunk_id="chk_test",
                title="Protein Note",
                page_or_section="section unavailable",
            )
        ],
        "safety_mode": "normal",
        "safety_flags": [],
        "errors": [],
        "timings": {},
    }

    result = node(state)

    assert result["insufficient_evidence"] is False
    assert "I found approved evidence" in result["draft_answer"]
    assert "[src_test / chk_test]" in result["draft_answer"]
    assert "simulated chat quota error" in result["errors"][0]
