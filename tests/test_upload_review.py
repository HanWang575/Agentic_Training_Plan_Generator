"""Upload and review behavior for the evidence knowledge base."""

from pathlib import Path

from sqlmodel import Session, select

from app.core.config import Settings
from app.knowledge.indexing import KeywordIndexer
from app.knowledge.service import KnowledgeService
from app.storage.database import create_app_engine, init_db
from app.storage.models import Citation, KnowledgeChunk, SourceDocument


def _make_service(tmp_path: Path):
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
    return settings, session, KnowledgeService(session=session, settings=settings)


def test_upload_creates_pending_source_and_chunks(tmp_path: Path):
    """A text upload is parsed and saved as pending_review, not approved."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "protein.txt"
    source_path.write_text("Protein supports muscle repair after resistance training.", encoding="utf-8")

    result = service.upload_file(source_path, title="Protein Note", topic_tags=["nutrition"])

    assert result.approval_status == "pending_review"
    assert result.parser_status == "parsed"
    assert result.chunk_count == 1
    session.close()


def test_create_pending_upload_returns_queued_source_before_parsing(tmp_path: Path):
    """Async upload registration should return immediately with queued parser state."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "queued.txt"
    source_path.write_text("Queued upload test.", encoding="utf-8")

    source_id = service.create_pending_upload(source_path, title="Queued Note", topic_tags=["test"])
    source = session.get(SourceDocument, source_id)

    assert source is not None
    assert source.parser_status == "queued"
    assert source.parser_progress == 0.0
    assert source.parser_progress_message == "Queued for parsing."
    session.close()


def test_parse_pending_source_updates_progress_and_creates_chunks(tmp_path: Path):
    """Queued uploads can be parsed later by the background worker path."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "later.txt"
    source_path.write_text("Protein supports muscle repair after training.", encoding="utf-8")

    source_id = service.create_pending_upload(source_path, title="Later Note", topic_tags=["nutrition"])
    result = service.parse_pending_source(source_id)
    source = session.get(SourceDocument, source_id)

    assert result.parser_status == "parsed"
    assert source is not None
    assert source.parser_progress == 1.0
    assert source.parser_progress_message == "Parsing complete."
    assert session.exec(select(KnowledgeChunk).where(KnowledgeChunk.source_id == source_id)).all()
    session.close()


def test_get_chunks_by_ids_returns_full_text_and_missing_ids(tmp_path: Path):
    """Chunk lookup returns readable text and reports IDs that are not stored."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "protein.txt"
    source_path.write_text("Protein supports muscle repair after resistance training.", encoding="utf-8")
    upload = service.upload_file(source_path, title="Protein Note", topic_tags=["nutrition"])
    chunk = session.exec(select(KnowledgeChunk).where(KnowledgeChunk.source_id == upload.source_id)).one()

    result = service.get_chunks_by_ids([chunk.chunk_id, "chk_missing"])

    assert result.missing_chunk_ids == ["chk_missing"]
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == chunk.chunk_id
    assert result.chunks[0].title == "Protein Note"
    assert result.chunks[0].topic_tags == ["nutrition"]
    assert "muscle repair" in result.chunks[0].text
    session.close()


def test_pending_source_is_not_searchable(tmp_path: Path):
    """Pending chunks remain excluded from approved-only keyword search."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "hyrox.txt"
    source_path.write_text("HYROX combines running with functional fitness stations.", encoding="utf-8")
    service.upload_file(source_path, title="HYROX Note", topic_tags=["hyrox"])

    keyword, semantic, semantic_error = service.search("HYROX")

    assert keyword == []
    assert semantic == []
    assert "OPENAI_API_KEY" in semantic_error
    session.close()


def test_approval_returns_index_error_when_semantic_config_is_missing(tmp_path: Path):
    """Approval does not create trusted knowledge unless both indexes can run."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "strength.txt"
    source_path.write_text("Progressive overload is a common strength training principle.", encoding="utf-8")
    upload = service.upload_file(source_path, title="Strength Note", topic_tags=["strength"])

    review = service.approve_source(upload.source_id)
    keyword, _, semantic_error = service.search("progressive overload")

    assert review.approval_status == "index_error"
    assert "semantic_index_config_error" in review.indexing_status
    assert "OPENAI_API_KEY" in review.message
    assert keyword == []
    assert "OPENAI_API_KEY" in semantic_error
    session.close()


def test_parse_error_source_cannot_be_approved(tmp_path: Path):
    """A failed parse remains visible and is not moved into approved knowledge."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "bad.bin"
    source_path.write_bytes(b"not a supported file")
    upload = service.upload_file(source_path, title="Bad Upload", topic_tags=[])

    review = service.approve_source(upload.source_id)

    assert upload.approval_status == "parse_error"
    assert review.approval_status == "parse_error"
    assert review.indexed_chunk_count == 0
    assert "cannot be approved" in review.message
    session.close()


def test_rag_corpus_upload_infers_title_and_folder_tags(tmp_path: Path):
    """Bulk upload uses file names as titles and corpus folders as tags."""

    _, session, service = _make_service(tmp_path)
    corpus_dir = tmp_path / "RAG_corpus"
    topic_dir = corpus_dir / "sports_nutrition"
    topic_dir.mkdir(parents=True)
    source_path = topic_dir / "daily_protein_note.txt"
    source_path.write_text("Protein needs can vary by training goal.", encoding="utf-8")
    skipped_path = topic_dir / "ignore.me"
    skipped_path.write_text("unsupported", encoding="utf-8")

    result = service.upload_rag_corpus(corpus_dir)

    assert result.uploaded_count == 1
    assert result.skipped_count == 1
    assert result.uploaded[0].title == "daily_protein_note"
    assert result.uploaded[0].topic_tags == ["sports_nutrition"]
    session.close()


def test_rag_corpus_plan_and_single_file_upload(tmp_path: Path):
    """The UI progress path can plan files and upload one relative corpus file."""

    _, session, service = _make_service(tmp_path)
    corpus_dir = tmp_path / "RAG_corpus"
    topic_dir = corpus_dir / "strength_powerlifting"
    topic_dir.mkdir(parents=True)
    source_path = topic_dir / "squat_note.txt"
    source_path.write_text("Squat training benefits from progressive overload.", encoding="utf-8")

    plan = service.list_rag_corpus_files(corpus_dir)
    upload = service.upload_one_rag_corpus_file("strength_powerlifting/squat_note.txt", corpus_dir)

    assert plan.file_count == 1
    assert plan.files[0].relative_path == "strength_powerlifting/squat_note.txt"
    assert upload.title == "squat_note"
    assert upload.topic_tags == ["strength_powerlifting"]
    session.close()


def test_remove_approved_source_deletes_database_rows_and_upload_files(tmp_path: Path):
    """Hard removal deletes approved source metadata, chunks, citations, and upload files."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "protein.txt"
    source_path.write_text("Protein supports muscle repair.", encoding="utf-8")
    upload = service.upload_file(source_path, title="Protein Note", topic_tags=["nutrition"])
    source = session.get(SourceDocument, upload.source_id)
    chunks = session.exec(
        select(KnowledgeChunk).where(KnowledgeChunk.source_id == upload.source_id)
    ).all()
    source.approval_status = "approved"
    for chunk in chunks:
        chunk.approval_status = "approved"
        session.add(chunk)
    session.add(
        Citation(
            citation_id="cit_test",
            trace_id="trc_test",
            source_id=upload.source_id,
            chunk_id=chunks[0].chunk_id,
            citation_text="Protein Note",
        )
    )
    session.add(source)
    session.commit()
    KeywordIndexer(session).index_source(upload.source_id)
    stored_dir = Path(source.file_path).parent

    result = service.remove_approved_sources(source_ids=[upload.source_id])

    assert result.removed_source_ids == [upload.source_id]
    assert session.get(SourceDocument, upload.source_id) is None
    assert session.exec(select(KnowledgeChunk).where(KnowledgeChunk.source_id == upload.source_id)).all() == []
    assert session.exec(select(Citation).where(Citation.source_id == upload.source_id)).all() == []
    assert not stored_dir.exists()
    keyword, _, _ = service.search("protein", keyword_top_k=5, semantic_top_k=1)
    assert keyword == []
    session.close()


def test_batch_review_rejects_selected_and_all_sources(tmp_path: Path):
    """Batch review supports selected sources and the review-all shortcut."""

    _, session, service = _make_service(tmp_path)
    source_ids = []
    for index in range(3):
        source_path = tmp_path / f"review_{index}.txt"
        source_path.write_text(f"review source {index}", encoding="utf-8")
        upload = service.upload_file(source_path, title=f"Review {index}", topic_tags=["test"])
        source_ids.append(upload.source_id)

    selected_result = service.review_sources(action="reject", source_ids=source_ids[:2])
    all_result = service.review_sources(action="reject", review_all=True)

    assert selected_result.processed_count == 2
    assert [item.source_id for item in selected_result.results] == source_ids[:2]
    assert all_result.processed_count == 1
    assert all_result.results[0].source_id == source_ids[2]
    assert session.get(SourceDocument, source_ids[0]).approval_status == "rejected"
    assert session.get(SourceDocument, source_ids[1]).approval_status == "rejected"
    assert session.get(SourceDocument, source_ids[2]).approval_status == "rejected"
    session.close()


def test_remove_rejected_source_records(tmp_path: Path):
    """Historical rejected sources can be hard-deleted from storage."""

    _, session, service = _make_service(tmp_path)
    source_path = tmp_path / "rejected.txt"
    source_path.write_text("This source will be rejected.", encoding="utf-8")
    upload = service.upload_file(source_path, title="Rejected", topic_tags=["test"])
    service.reject_source(upload.source_id)
    stored_dir = Path(session.get(SourceDocument, upload.source_id).file_path).parent

    result = service.remove_rejected_sources(remove_all=True)

    assert result.removed_source_ids == [upload.source_id]
    assert session.get(SourceDocument, upload.source_id) is None
    assert session.exec(select(KnowledgeChunk).where(KnowledgeChunk.source_id == upload.source_id)).all() == []
    assert not stored_dir.exists()
    session.close()


def test_remove_multiple_and_all_approved_sources(tmp_path: Path):
    """Removal supports a selected list and the all-approved shortcut."""

    _, session, service = _make_service(tmp_path)
    source_ids = []
    for index in range(3):
        source_path = tmp_path / f"note_{index}.txt"
        source_path.write_text(f"approved source {index}", encoding="utf-8")
        upload = service.upload_file(source_path, title=f"Note {index}", topic_tags=["test"])
        source = session.get(SourceDocument, upload.source_id)
        chunks = session.exec(
            select(KnowledgeChunk).where(KnowledgeChunk.source_id == upload.source_id)
        ).all()
        source.approval_status = "approved"
        for chunk in chunks:
            chunk.approval_status = "approved"
            session.add(chunk)
        session.add(source)
        source_ids.append(upload.source_id)
    session.commit()

    selected_result = service.remove_approved_sources(source_ids=source_ids[:2])
    all_result = service.remove_approved_sources(remove_all=True)

    assert selected_result.removed_source_ids == source_ids[:2]
    assert session.get(SourceDocument, source_ids[0]) is None
    assert session.get(SourceDocument, source_ids[1]) is None
    assert all_result.removed_source_ids == [source_ids[2]]
    assert session.get(SourceDocument, source_ids[2]) is None
    session.close()
