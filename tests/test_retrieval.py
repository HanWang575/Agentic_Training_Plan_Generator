"""Retrieval ranking helpers."""

from app.knowledge.schemas import RetrievedChunk
from app.knowledge.service import KnowledgeService
from app.qa.nodes import ResultMerger
from app.core.config import Settings
from app.storage.database import create_app_engine, init_db
from app.storage.models import KnowledgeChunk, SourceDocument

from sqlmodel import Session


def _chunk(chunk_id: str, rank: int, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        source_id="src_test",
        chunk_id=chunk_id,
        title="Test",
        text="example",
        page_number=None,
        section_heading=None,
        topic_tags=[],
        token_count=1,
        score=1.0 / rank,
        rank=rank,
        retrieval_method=method,
    )


def test_rrf_prefers_chunk_found_by_both_methods(tmp_path):
    """A chunk present in keyword and semantic lists receives combined RRF score."""

    settings = Settings(_env_file=None, sqlite_path=tmp_path / "app.sqlite", upload_dir=tmp_path / "uploads")
    merger = ResultMerger(settings)
    state = {
        "keyword_results": [_chunk("chk_shared", 2, "keyword"), _chunk("chk_keyword", 1, "keyword")],
        "semantic_results": [_chunk("chk_shared", 1, "semantic")],
        "timings": {},
    }

    result = merger(state)

    assert result["merged_results"][0].chunk_id == "chk_shared"


def test_neighbor_expansion_adds_previous_and_next_approved_chunks(tmp_path):
    """Final context includes one neighboring chunk before and after a hit."""

    settings = Settings(_env_file=None, sqlite_path=tmp_path / "app.sqlite", upload_dir=tmp_path / "uploads")
    engine = create_app_engine(settings)
    init_db(engine)
    session = Session(engine)
    source = SourceDocument(
        source_id="src_neighbor",
        title="Neighbor Source",
        source_type="txt",
        file_path="neighbor.txt",
        approval_status="approved",
        parser_status="parsed",
    )
    session.add(source)
    for index in range(3):
        session.add(
            KnowledgeChunk(
                chunk_id=f"chk_neighbor_{index}",
                source_id="src_neighbor",
                chunk_index=index,
                text=f"chunk {index}",
                token_count=2,
                approval_status="approved",
            )
        )
    session.commit()
    service = KnowledgeService(session=session, settings=settings)

    expanded = service.add_neighbor_chunks([_chunk("chk_neighbor_1", 1, "keyword")])

    assert [chunk.chunk_id for chunk in expanded] == [
        "chk_neighbor_0",
        "chk_neighbor_1",
        "chk_neighbor_2",
    ]
    session.close()
