"""FastAPI routes for upload, review, source listing, and search."""

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.knowledge.schemas import (
    BatchReviewRequest,
    BatchReviewResult,
    BulkUploadResult,
    ChunkLookupRequest,
    ChunkLookupResponse,
    CorpusPlanResult,
    RemoveSourcesRequest,
    RemoveSourcesResult,
    ReviewResult,
    SearchKnowledgeRequest,
    SearchKnowledgeResponse,
    SourceSummary,
    UploadResult,
)
from app.knowledge.service import KnowledgeService
from app.storage.database import engine, get_session

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


def _service_dependency():
    settings = get_settings()
    for session in get_session():
        yield KnowledgeService(session=session, settings=settings)


def _parse_upload_in_background(source_id: str) -> None:
    """Parse one queued upload after the async route returns."""

    settings = get_settings()
    with Session(engine) as session:
        KnowledgeService(session=session, settings=settings).parse_pending_source(source_id)


@router.post("/upload", response_model=UploadResult)
def upload_source(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    topic_tags: str | None = Form(default=None),
    service: KnowledgeService = Depends(_service_dependency),
    settings: Settings = Depends(get_settings),
) -> UploadResult:
    """Upload one source file and store parsed chunks as pending review."""

    original_path = Path(file.filename or "upload")
    suffix = original_path.suffix
    inferred_title = title.strip() if title and title.strip() else original_path.stem
    tags = [tag.strip() for tag in (topic_tags or "").split(",") if tag.strip()]
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)
    try:
        if temp_path.stat().st_size > settings.maximum_upload_bytes:
            raise HTTPException(status_code=413, detail="File is larger than the 100 MB v1 limit.")
        return service.upload_file(temp_path, title=inferred_title, topic_tags=tags)
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/upload/async", response_model=UploadResult)
def upload_source_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    topic_tags: str | None = Form(default=None),
    service: KnowledgeService = Depends(_service_dependency),
    settings: Settings = Depends(get_settings),
) -> UploadResult:
    """Upload one source and parse it in the background for UI progress polling."""

    original_path = Path(file.filename or "upload")
    suffix = original_path.suffix
    inferred_title = title.strip() if title and title.strip() else original_path.stem
    tags = [tag.strip() for tag in (topic_tags or "").split(",") if tag.strip()]
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)
    try:
        if temp_path.stat().st_size > settings.maximum_upload_bytes:
            raise HTTPException(status_code=413, detail="File is larger than the 100 MB v1 limit.")
        source_id = service.create_pending_upload(temp_path, title=inferred_title, topic_tags=tags)
        background_tasks.add_task(_parse_upload_in_background, source_id)
        return service.get_source_summary(source_id, result_type=UploadResult)
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/upload/rag-corpus", response_model=BulkUploadResult)
def upload_rag_corpus(
    service: KnowledgeService = Depends(_service_dependency),
    settings: Settings = Depends(get_settings),
) -> BulkUploadResult:
    """Upload all supported files from the configured `RAG_corpus` folder."""

    return service.upload_rag_corpus(settings.rag_corpus_dir)


@router.get("/upload/rag-corpus/plan", response_model=CorpusPlanResult)
def plan_rag_corpus_upload(
    service: KnowledgeService = Depends(_service_dependency),
    settings: Settings = Depends(get_settings),
) -> CorpusPlanResult:
    """Preview supported files under `RAG_corpus` before uploading."""

    return service.list_rag_corpus_files(settings.rag_corpus_dir)


@router.post("/upload/rag-corpus/file", response_model=UploadResult)
def upload_one_rag_corpus_file(
    relative_path: str,
    service: KnowledgeService = Depends(_service_dependency),
    settings: Settings = Depends(get_settings),
) -> UploadResult:
    """Upload one supported file from `RAG_corpus` by relative path."""

    try:
        return service.upload_one_rag_corpus_file(relative_path, settings.rag_corpus_dir)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/sources", response_model=list[SourceSummary])
def list_sources(
    status: str | None = None,
    service: KnowledgeService = Depends(_service_dependency),
) -> list[SourceSummary]:
    """List source summaries, optionally filtered by approval status."""

    return service.list_sources(status=status)


@router.get("/sources/{source_id}", response_model=SourceSummary)
def get_source(
    source_id: str,
    service: KnowledgeService = Depends(_service_dependency),
) -> SourceSummary:
    """Return one source summary for polling upload progress."""

    try:
        return service.get_source_summary(source_id, result_type=SourceSummary)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/chunks", response_model=ChunkLookupResponse)
def get_chunks(
    request: ChunkLookupRequest,
    service: KnowledgeService = Depends(_service_dependency),
) -> ChunkLookupResponse:
    """Return full stored text for requested chunk IDs."""

    return service.get_chunks_by_ids(request.chunk_ids)


@router.post("/review/batch", response_model=BatchReviewResult)
def batch_review_sources(
    request: BatchReviewRequest,
    service: KnowledgeService = Depends(_service_dependency),
) -> BatchReviewResult:
    """Approve or reject selected/all reviewable sources."""

    try:
        return service.review_sources(
            action=request.action,
            source_ids=request.source_ids,
            review_all=request.review_all,
            review_note=request.review_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sources/remove", response_model=RemoveSourcesResult)
def remove_approved_sources(
    request: RemoveSourcesRequest,
    service: KnowledgeService = Depends(_service_dependency),
) -> RemoveSourcesResult:
    """Remove single, multiple, or all approved sources from the knowledge base."""

    return service.remove_approved_sources(
        source_ids=request.source_ids,
        remove_all=request.remove_all,
    )


@router.post("/sources/rejected/remove", response_model=RemoveSourcesResult)
def remove_rejected_sources(
    request: RemoveSourcesRequest,
    service: KnowledgeService = Depends(_service_dependency),
) -> RemoveSourcesResult:
    """Remove historical rejected source records."""

    return service.remove_rejected_sources(
        source_ids=request.source_ids,
        remove_all=request.remove_all,
    )


@router.delete("/sources/{source_id}", response_model=RemoveSourcesResult)
def remove_one_approved_source(
    source_id: str,
    service: KnowledgeService = Depends(_service_dependency),
) -> RemoveSourcesResult:
    """Remove one approved source from the knowledge base."""

    return service.remove_approved_sources(source_ids=[source_id])


@router.post("/{source_id}/approve", response_model=ReviewResult)
def approve_source(
    source_id: str,
    review_note: str | None = None,
    service: KnowledgeService = Depends(_service_dependency),
) -> ReviewResult:
    """Approve a source and index approved chunks."""

    try:
        return service.approve_source(source_id=source_id, review_note=review_note)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{source_id}/reject", response_model=ReviewResult)
def reject_source(
    source_id: str,
    review_note: str | None = None,
    service: KnowledgeService = Depends(_service_dependency),
) -> ReviewResult:
    """Reject a source and exclude it from search."""

    try:
        return service.reject_source(source_id=source_id, review_note=review_note)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/search", response_model=SearchKnowledgeResponse)
def search_knowledge(
    request: SearchKnowledgeRequest,
    service: KnowledgeService = Depends(_service_dependency),
) -> SearchKnowledgeResponse:
    """Search approved knowledge chunks without generating an answer."""

    keyword_results, semantic_results, semantic_error = service.search(
        request.query,
        topic_hint=request.topic_hint,
        keyword_top_k=request.top_k,
        semantic_top_k=request.top_k,
    )
    chunks = [
        {
            "source_id": chunk.source_id,
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "page_number": chunk.page_number,
            "section_heading": chunk.section_heading,
            "content_type": chunk.content_type,
            "content_metadata": chunk.content_metadata,
            "score": chunk.score,
            "retrieval_method": chunk.retrieval_method,
            "text_preview": chunk.text[:600],
        }
        for chunk in [*keyword_results, *semantic_results]
    ]
    if semantic_error:
        chunks.append({"semantic_error": semantic_error})
    return SearchKnowledgeResponse(chunks=chunks)
