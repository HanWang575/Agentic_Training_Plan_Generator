"""Q&A workflow behavior."""

from pathlib import Path

from sqlmodel import Session

from app.core.config import Settings
from app.core.llm_provider import QueryPlanResult
from app.knowledge.service import KnowledgeService
from app.qa.nodes import AnswerGenerator, CitationChecker, QueryPlanner
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
    assert trace.merged_chunk_ids == []
    assert trace.final_chunk_ids == []
    session.close()


def test_safety_question_adds_safety_note(tmp_path: Path):
    """Injury, medical, doping, and extreme diet terms are visible in safety output."""

    session, _, qa = _make_services(tmp_path)

    response = qa.ask("Can I train through knee injury pain?", debug=False)

    assert response.safety_note is not None
    assert "injury" in response.safety_note
    session.close()


def test_trace_persists_query_planner_fields(tmp_path: Path):
    """Stored traces should keep planner output for later UI inspection."""

    class FakePlannerProvider:
        def generate_query_plan(self, request):
            return QueryPlanResult(
                clean_question="What does VO2max mean?",
                clarified_question="Explain what VO2max means in endurance training.",
                search_query="VO2max maximal oxygen uptake endurance training aerobic capacity",
                query_terms=["VO2max", "maximal oxygen uptake", "endurance training"],
                notes=["Expanded VO2max to maximal oxygen uptake."],
            )

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
    provider = FakePlannerProvider()
    knowledge = KnowledgeService(session=session, settings=settings, llm_provider=provider)
    qa = QAService(session=session, settings=settings, knowledge_service=knowledge, llm_provider=provider)

    response = qa.ask("What does VO2max mean?", topic_hint="running", debug=True)
    trace = qa.get_trace(response.trace_id)

    assert response.debug["planner_used"] == "llm"
    assert trace.planner_used == "llm"
    assert trace.clean_question == "What does VO2max mean?"
    assert trace.clarified_question == "Explain what VO2max means in endurance training."
    assert trace.search_query == "VO2max maximal oxygen uptake endurance training aerobic capacity"
    assert trace.query_terms == [
        "Explain what VO2max means in endurance training.",
        "VO2max",
        "maximal oxygen uptake",
        "endurance training",
    ]
    assert trace.planner_notes == ["Expanded VO2max to maximal oxygen uptake."]
    session.close()


def test_llm_query_planner_rewrites_and_expands_query():
    """The LLM planner should clean, clarify, and expand retrieval input."""

    class FakePlannerProvider:
        def generate_query_plan(self, request):
            return QueryPlanResult(
                clean_question="VO2max 是什么意思",
                clarified_question="解释 VO2max 的含义，以及它在耐力训练中的意义",
                search_query="VO2max maximal oxygen uptake endurance training aerobic capacity",
                query_terms=["VO2max", "maximal oxygen uptake", "endurance training", "aerobic capacity"],
                notes=["Expanded VO2max to maximal oxygen uptake."],
            )

    planner = QueryPlanner(FakePlannerProvider())
    result = planner(
        {
            "question": "VO2max 是什么意思",
            "clean_question": "VO2max 是什么意思",
            "topic_hint": "running",
            "errors": [],
            "timings": {},
        }
    )

    assert result["planner_used"] == "llm"
    assert result["clean_question"] == "VO2max 是什么意思"
    assert result["clarified_question"] == "解释 VO2max 的含义，以及它在耐力训练中的意义"
    assert "maximal oxygen uptake" in result["search_query"]
    assert "endurance training" in result["query_terms"]
    assert result["planner_notes"] == ["Expanded VO2max to maximal oxygen uptake."]


def test_query_planner_falls_back_when_llm_planner_fails():
    """Planning should degrade gracefully when the planner LLM is unavailable."""

    class FailingPlannerProvider:
        def generate_query_plan(self, request):
            raise RuntimeError("planner timeout")

    planner = QueryPlanner(FailingPlannerProvider())
    result = planner(
        {
            "question": "How much protein should I eat?",
            "clean_question": "How much protein should I eat?",
            "topic_hint": "nutrition",
            "errors": [],
            "timings": {},
        }
    )

    assert result["planner_used"] == "fallback"
    assert result["clean_question"] == "How much protein should I eat?"
    assert result["clarified_question"] == "How much protein should I eat?"
    assert "amino acids" in result["query_terms"]
    assert any("planner timeout" in item for item in result["planner_notes"])
    assert any("Query planner fallback used" in item for item in result["errors"])


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


def test_citation_checker_moves_numeric_citations_before_punctuation():
    """Inline numeric citations should read naturally before sentence punctuation."""

    checker = CitationChecker()
    result = checker(
        {
            "draft_answer": (
                "Short answer:\n"
                "The template needs restraint with assistance work.[2]\n\n"
                "Evidence used:\n"
                "2. [src_a / chk_a] Strength Template, page 4"
            ),
            "citations": [
                CitationOut(
                    source_id="src_a",
                    chunk_id="chk_a",
                    title="Strength Template",
                    page_or_section="page 4",
                )
            ],
            "insufficient_evidence": False,
            "errors": [],
            "timings": {},
        }
    )

    assert "assistance work [2]." in result["final_answer"]
    assert "assistance work.[2]" not in result["final_answer"]


def test_citation_checker_accepts_citations_from_final_context_chunks():
    """A marker is supported when its chunk was actually sent to the LLM."""

    checker = CitationChecker()
    result = checker(
        {
            "draft_answer": (
                "Short answer:\n"
                "VO2max is a measure of maximal oxygen uptake.\n\n"
                "Evidence used:\n"
                "- [src_ctx / chk_ctx] Daniels Running Formula, page 71"
            ),
            "citations": [],
            "context_results": [
                RetrievedChunk(
                    source_id="src_ctx",
                    chunk_id="chk_ctx",
                    title="Daniels Running Formula",
                    text="VO2max means maximal oxygen uptake.",
                    page_number=71,
                    section_heading=None,
                    topic_tags=["running"],
                    token_count=6,
                    score=1.0,
                    rank=1,
                    retrieval_method="neighbor",
                )
            ],
            "insufficient_evidence": False,
            "errors": [],
            "timings": {},
        }
    )

    assert "[unsupported citation removed]" not in result["final_answer"]
    assert result["citations"] == [
        CitationOut(
            source_id="src_ctx",
            chunk_id="chk_ctx",
            title="Daniels Running Formula",
            page_or_section="page 71",
        )
    ]


def test_citation_checker_repairs_wrong_chunk_id_from_matching_final_context_page():
    """Wrong chunk IDs are rewritten when title/page clearly match a final chunk."""

    checker = CitationChecker()
    result = checker(
        {
            "draft_answer": (
                "Evidence used:\n"
                "- [src_ctx / chk_wrong] Daniels Running Formula 3rd Edition, page 71"
            ),
            "citations": [],
            "context_results": [
                RetrievedChunk(
                    source_id="src_ctx",
                    chunk_id="chk_right",
                    title="Daniels Running Formula 3rd Edition",
                    text="VO2max means maximal oxygen uptake.",
                    page_number=71,
                    section_heading=None,
                    topic_tags=["running"],
                    token_count=6,
                    score=1.0,
                    rank=1,
                    retrieval_method="neighbor",
                )
            ],
            "insufficient_evidence": False,
            "errors": [],
            "timings": {},
        }
    )

    assert "[unsupported citation removed]" not in result["final_answer"]
    assert "[src_ctx / chk_right]" in result["final_answer"]
    assert "[src_ctx / chk_wrong]" not in result["final_answer"]
    assert result["citations"][0].chunk_id == "chk_right"


def test_citation_checker_repairs_wrong_chunk_id_from_matching_final_context_page_range():
    """A model line like page 61-62 can be repaired to a final chunk on page 62."""

    checker = CitationChecker()
    result = checker(
        {
            "draft_answer": (
                "Evidence used:\n"
                "- [src_ctx / chk_wrong] Daniels Running Formula 3rd Edition, page 61-62"
            ),
            "citations": [],
            "context_results": [
                RetrievedChunk(
                    source_id="src_ctx",
                    chunk_id="chk_page62",
                    title="Daniels Running Formula 3rd Edition",
                    text="Running economy and oxygen uptake are discussed here.",
                    page_number=62,
                    section_heading=None,
                    topic_tags=["running"],
                    token_count=7,
                    score=1.0,
                    rank=1,
                    retrieval_method="neighbor",
                )
            ],
            "insufficient_evidence": False,
            "errors": [],
            "timings": {},
        }
    )

    assert "[unsupported citation removed]" not in result["final_answer"]
    assert "[src_ctx / chk_page62]" in result["final_answer"]
    assert result["citations"][0].chunk_id == "chk_page62"


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
    assert any("Semantic retrieval failed" in error for error in response.debug["errors"])
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
