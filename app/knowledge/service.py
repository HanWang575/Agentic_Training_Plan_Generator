"""Service layer for upload, review, indexing, source listing, and search."""

import json
import shutil
from pathlib import Path
from uuid import uuid4

from sqlmodel import Session, func, select

from app.core.config import Settings
from app.core.errors import ConfigurationError, NotFoundError
from app.core.llm_provider import OpenAIProvider
from app.knowledge.chunking import chunk_document
from app.knowledge.indexing import KeywordIndexer, VectorIndexer
from app.knowledge.parsers import parse_document
from app.knowledge.retrieval import KeywordRetriever, SemanticRetriever, normalize_query, safe_semantic_search
from app.knowledge.schemas import (
    BatchReviewResult,
    BulkUploadResult,
    CorpusFileItem,
    CorpusPlanResult,
    RemoveSourcesResult,
    RetrievedChunk,
    ReviewResult,
    SUPPORTED_FILE_TYPES,
    SourceSummary,
    UploadResult,
)
from app.storage.models import Citation, KnowledgeChunk, SourceDocument, utc_now


class KnowledgeService:
    """Stable service interface for the evidence knowledge base."""

    def __init__(
        self,
        session: Session,
        settings: Settings,
        llm_provider: OpenAIProvider | None = None,
    ):
        self.session = session
        self.settings = settings
        self.llm_provider = llm_provider or OpenAIProvider(settings)

    def upload_file(
        self,
        source_file_path: Path,
        title: str | None,
        topic_tags: list[str] | None,
    ) -> UploadResult:
        """Parse and store an uploaded file as pending review.

        Parameters:
            source_file_path: temporary file path from FastAPI or local caller.
            title: optional user-supplied source title.
            topic_tags: optional list with shape `[tag_count]`.
        Returns:
            UploadResult: source metadata and chunk count.
        """

        source_id = f"src_{uuid4().hex[:12]}"
        tags = topic_tags or []
        source_dir = self.settings.upload_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        stored_path = source_dir / source_file_path.name
        shutil.copy2(source_file_path, stored_path)

        source = SourceDocument(
            source_id=source_id,
            title=title or source_file_path.stem,
            source_type=source_file_path.suffix.lower().lstrip("."),
            file_path=str(stored_path),
            topic_tags_json=json.dumps(tags),
            approval_status="pending_review",
        )
        self.session.add(source)
        self.session.commit()

        try:
            parsed = parse_document(stored_path, title=source.title)
            drafts = chunk_document(
                parsed,
                target_tokens=self.settings.chunk_target_tokens,
                overlap_tokens=self.settings.chunk_overlap_tokens,
            )
            source.parser_status = "parsed"
            for draft in drafts:
                self.session.add(
                    KnowledgeChunk(
                        chunk_id=f"chk_{uuid4().hex[:16]}",
                        source_id=source_id,
                        chunk_index=draft.chunk_index,
                        text=draft.text,
                        page_number=draft.page_number,
                        section_heading=draft.section_heading,
                        topic_tags_json=json.dumps(tags),
                        token_count=draft.token_count,
                        approval_status="pending_review",
                    )
                )
            self.session.add(source)
            self.session.commit()
        except Exception as exc:  # Store parser errors so review UI can show them.
            source.approval_status = "parse_error"
            source.parser_status = "parse_error"
            source.parser_error = str(exc)
            self.session.add(source)
            self.session.commit()

        return self._summary_for_source(source_id, result_type=UploadResult)

    def upload_rag_corpus(self, corpus_dir: Path | None = None) -> BulkUploadResult:
        """Upload every supported source file from a local RAG corpus folder.

        Parameters:
            corpus_dir: root folder to scan. Defaults to `settings.rag_corpus_dir`.
        Returns:
            BulkUploadResult: uploaded source summaries plus skipped files.

        Titles are inferred from file names. Topic tags are inferred from parent
        folders below the corpus root, e.g. `RAG_corpus/sports_nutrition/a.pdf`
        becomes title `a` and tag `sports_nutrition`.
        """

        root = (corpus_dir or self.settings.rag_corpus_dir).resolve()
        plan = self.list_rag_corpus_files(root)
        uploaded: list[UploadResult] = []
        skipped = list(plan.skipped)
        for item in plan.files:
            try:
                uploaded.append(self.upload_one_rag_corpus_file(item.relative_path, root))
            except Exception as exc:
                skipped.append({"path": item.relative_path, "reason": str(exc)})

        return BulkUploadResult(
            uploaded=uploaded,
            skipped=skipped,
            uploaded_count=len(uploaded),
            skipped_count=len(skipped),
        )

    def list_rag_corpus_files(self, corpus_dir: Path | None = None) -> CorpusPlanResult:
        """List supported files under the RAG corpus before uploading them.

        Parameters:
            corpus_dir: root folder to scan. Defaults to `settings.rag_corpus_dir`.
        Returns:
            CorpusPlanResult: uploadable file metadata plus skipped files.
        """

        root = (corpus_dir or self.settings.rag_corpus_dir).resolve()
        if not root.exists() or not root.is_dir():
            return CorpusPlanResult(
                skipped=[{"path": str(root), "reason": "RAG corpus folder was not found."}],
                skipped_count=1,
            )

        files: list[CorpusFileItem] = []
        skipped: list[dict] = []
        for source_path in sorted(path for path in root.rglob("*") if path.is_file()):
            relative_path = source_path.relative_to(root).as_posix()
            if source_path.suffix.lower() not in SUPPORTED_FILE_TYPES:
                skipped.append({"path": relative_path, "reason": "unsupported file type"})
                continue
            if source_path.stat().st_size > self.settings.maximum_upload_bytes:
                skipped.append({"path": relative_path, "reason": "file exceeds upload size limit"})
                continue
            files.append(
                CorpusFileItem(
                    relative_path=relative_path,
                    title=_title_from_file_name(source_path),
                    topic_tags=_tags_from_corpus_path(source_path, root),
                    source_type=source_path.suffix.lower().lstrip("."),
                    size_bytes=source_path.stat().st_size,
                )
            )
        return CorpusPlanResult(
            files=files,
            skipped=skipped,
            file_count=len(files),
            skipped_count=len(skipped),
        )

    def upload_one_rag_corpus_file(
        self,
        relative_path: str,
        corpus_dir: Path | None = None,
    ) -> UploadResult:
        """Upload one supported source from `RAG_corpus`.

        Parameters:
            relative_path: path below the corpus root, never absolute.
            corpus_dir: root folder to scan. Defaults to `settings.rag_corpus_dir`.
        Returns:
            UploadResult: parsed pending source summary.
        """

        root = (corpus_dir or self.settings.rag_corpus_dir).resolve()
        source_path = _resolve_corpus_relative_path(root, relative_path)
        if not source_path.exists() or not source_path.is_file():
            raise NotFoundError(f"RAG corpus file not found: {relative_path}")
        if source_path.suffix.lower() not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {source_path.suffix}")
        if source_path.stat().st_size > self.settings.maximum_upload_bytes:
            raise ValueError("File exceeds upload size limit.")
        return self.upload_file(
            source_path,
            title=_title_from_file_name(source_path),
            topic_tags=_tags_from_corpus_path(source_path, root),
        )

    def approve_source(self, source_id: str, review_note: str | None = None) -> ReviewResult:
        """Approve a source and index its chunks.

        Parameters:
            source_id: source ID with shape `src_*`.
            review_note: optional human review note.
        Returns:
            ReviewResult: final approval and indexing status.
        """

        source = self._get_source(source_id)
        if source.approval_status == "parse_error" or source.parser_status == "parse_error":
            return ReviewResult(
                source_id=source_id,
                approval_status=source.approval_status,
                indexing_status="not_indexed",
                indexed_chunk_count=0,
                message="Source has a parser error and cannot be approved until it is uploaded again.",
            )
        chunks = self.session.exec(
            select(KnowledgeChunk).where(KnowledgeChunk.source_id == source_id)
        ).all()
        source.approval_status = "approved"
        source.reviewed_at = utc_now()
        source.review_note = review_note
        for chunk in chunks:
            chunk.approval_status = "approved"
            self.session.add(chunk)
        self.session.add(source)
        self.session.commit()

        keyword_count = KeywordIndexer(self.session).index_source(source_id)
        try:
            vector_indexer = VectorIndexer(self.settings, self.llm_provider)
            semantic_count = vector_indexer.index_chunks(source, chunks)
        except ConfigurationError as exc:
            self._remove_semantic_index_rows(source_id)
            return self._mark_source_index_error(
                source=source,
                chunks=chunks,
                source_id=source_id,
                keyword_count=keyword_count,
                review_note=review_note,
                indexing_status="semantic_index_config_error",
                message=str(exc),
            )
        except Exception as exc:
            self._remove_semantic_index_rows(source_id)
            return self._mark_source_index_error(
                source=source,
                chunks=chunks,
                source_id=source_id,
                keyword_count=keyword_count,
                review_note=review_note,
                indexing_status="semantic_index_error",
                message=str(exc),
            )
        indexing_status = f"keyword_indexed:{keyword_count}; semantic_indexed:{semantic_count}"
        return ReviewResult(
            source_id=source_id,
            approval_status=source.approval_status,
            indexing_status=indexing_status,
            indexed_chunk_count=keyword_count,
            message=None,
        )

    def _mark_source_index_error(
        self,
        source: SourceDocument,
        chunks: list[KnowledgeChunk],
        source_id: str,
        keyword_count: int,
        review_note: str | None,
        indexing_status: str,
        message: str,
    ) -> ReviewResult:
        """Move a source back out of approved knowledge after indexing fails."""

        for chunk in chunks:
            chunk.approval_status = "index_error"
            self.session.add(chunk)
        source.approval_status = "index_error"
        source.review_note = review_note
        self.session.add(source)
        KeywordIndexer(self.session).remove_source(source_id)
        self.session.commit()
        return ReviewResult(
            source_id=source_id,
            approval_status="index_error",
            indexing_status=f"keyword_indexed:{keyword_count}; {indexing_status}",
            indexed_chunk_count=0,
            message=message,
        )

    def _remove_semantic_index_rows(self, source_id: str) -> None:
        """Best-effort cleanup for partial Chroma writes after failed indexing."""

        try:
            VectorIndexer(self.settings, self.llm_provider).remove_source(source_id)
        except Exception:
            pass

    def reject_source(self, source_id: str, review_note: str | None = None) -> ReviewResult:
        """Reject a source and remove it from keyword search."""

        source = self._get_source(source_id)
        source.approval_status = "rejected"
        source.reviewed_at = utc_now()
        source.review_note = review_note
        chunks = self.session.exec(
            select(KnowledgeChunk).where(KnowledgeChunk.source_id == source_id)
        ).all()
        for chunk in chunks:
            chunk.approval_status = "rejected"
            self.session.add(chunk)
        self.session.add(source)
        KeywordIndexer(self.session).remove_source(source_id)
        self.session.commit()
        return ReviewResult(
            source_id=source_id,
            approval_status="rejected",
            indexing_status="removed_from_keyword_index",
        )

    def review_sources(
        self,
        action: str,
        source_ids: list[str] | None = None,
        review_all: bool = False,
        review_note: str | None = None,
    ) -> BatchReviewResult:
        """Approve or reject selected/all reviewable sources.

        Parameters:
            action: `approve` or `reject`.
            source_ids: selected source IDs with shape `[source_count]`.
            review_all: when true, process all pending/index-error sources.
            review_note: optional note applied to every source.
        Returns:
            BatchReviewResult: per-source review results and failures.
        """

        normalized_action = action.strip().lower()
        if normalized_action not in {"approve", "reject"}:
            raise ValueError("Batch review action must be 'approve' or 'reject'.")

        ids = list(dict.fromkeys(source_ids or []))
        if review_all:
            sources = self.session.exec(
                select(SourceDocument).where(
                    SourceDocument.approval_status.in_(["pending_review", "index_error"])
                )
            ).all()
            ids = [source.source_id for source in sources]

        results: list[ReviewResult] = []
        failed: list[dict] = []
        for source_id in ids:
            try:
                if normalized_action == "approve":
                    results.append(self.approve_source(source_id, review_note))
                else:
                    results.append(self.reject_source(source_id, review_note))
            except Exception as exc:
                failed.append({"source_id": source_id, "error": str(exc)})
        return BatchReviewResult(
            action=normalized_action,
            results=results,
            failed=failed,
            processed_count=len(results),
            failed_count=len(failed),
        )

    def remove_approved_sources(
        self,
        source_ids: list[str] | None = None,
        remove_all: bool = False,
    ) -> RemoveSourcesResult:
        """Hard-delete approved sources from the knowledge base.

        Parameters:
            source_ids: optional list with shape `[source_count]` for single or
                multiple approved sources.
            remove_all: when true, remove all approved sources and ignore
                `source_ids`.
        Returns:
            RemoveSourcesResult: removed IDs plus requested IDs that were not
            currently approved sources.
        """

        return self._remove_sources_by_status(
            allowed_statuses=["approved"],
            source_ids=source_ids,
            remove_all=remove_all,
            label="approved",
        )

    def remove_rejected_sources(
        self,
        source_ids: list[str] | None = None,
        remove_all: bool = False,
    ) -> RemoveSourcesResult:
        """Hard-delete historical rejected source records.

        Parameters:
            source_ids: optional list with shape `[source_count]`.
            remove_all: when true, remove all `rejected` sources.
        Returns:
            RemoveSourcesResult: deleted rejected source IDs and misses.
        """

        return self._remove_sources_by_status(
            allowed_statuses=["rejected"],
            source_ids=source_ids,
            remove_all=remove_all,
            label="rejected",
        )

    def _remove_sources_by_status(
        self,
        allowed_statuses: list[str],
        source_ids: list[str] | None,
        remove_all: bool,
        label: str,
    ) -> RemoveSourcesResult:
        """Hard-delete sources whose approval status is in `allowed_statuses`."""

        requested_ids = list(dict.fromkeys(source_ids or []))
        if remove_all:
            sources = self.session.exec(
                select(SourceDocument).where(SourceDocument.approval_status.in_(allowed_statuses))
            ).all()
            requested_ids = [source.source_id for source in sources]
        elif requested_ids:
            sources = self.session.exec(
                select(SourceDocument).where(
                    SourceDocument.source_id.in_(requested_ids),
                    SourceDocument.approval_status.in_(allowed_statuses),
                )
            ).all()
        else:
            return RemoveSourcesResult(message="No source IDs were provided.")

        sources_by_id = {source.source_id: source for source in sources}
        removed_ids: list[str] = []
        for source_id in requested_ids:
            source = sources_by_id.get(source_id)
            if source is None:
                continue
            self._remove_source_indexes(source_id)
            self._delete_source_database_rows(source_id)
            self._delete_uploaded_source_files(source)
            removed_ids.append(source_id)

        self.session.commit()
        removed_set = set(removed_ids)
        missing_ids = [source_id for source_id in requested_ids if source_id not in removed_set]
        return RemoveSourcesResult(
            removed_source_ids=removed_ids,
            missing_source_ids=missing_ids,
            removed_count=len(removed_ids),
            message=f"Removed {len(removed_ids)} {label} source record(s).",
        )

    def _remove_source_indexes(self, source_id: str) -> None:
        """Remove one source from keyword and vector indexes."""

        KeywordIndexer(self.session).remove_source(source_id)
        try:
            VectorIndexer(self.settings, self.llm_provider).remove_source(source_id)
        except Exception:
            pass

    def _delete_source_database_rows(self, source_id: str) -> None:
        """Delete source, chunk, and citation rows for one approved source."""

        citations = self.session.exec(select(Citation).where(Citation.source_id == source_id)).all()
        chunks = self.session.exec(select(KnowledgeChunk).where(KnowledgeChunk.source_id == source_id)).all()
        source = self.session.get(SourceDocument, source_id)
        for citation in citations:
            self.session.delete(citation)
        for chunk in chunks:
            self.session.delete(chunk)
        if source is not None:
            self.session.delete(source)

    def _delete_uploaded_source_files(self, source: SourceDocument) -> None:
        """Delete the copied upload folder for one source, staying inside upload_dir."""

        try:
            stored_path = Path(source.file_path).resolve()
            upload_root = self.settings.upload_dir.resolve()
        except OSError:
            return
        if upload_root not in stored_path.parents:
            return
        source_dir = stored_path.parent
        if source_dir == upload_root or upload_root not in source_dir.parents:
            stored_path.unlink(missing_ok=True)
            return
        shutil.rmtree(source_dir, ignore_errors=True)

    def list_sources(self, status: str | None = None) -> list[SourceSummary]:
        """List sources, optionally filtered by approval status."""

        statement = select(SourceDocument)
        if status:
            statement = statement.where(SourceDocument.approval_status == status)
        sources = self.session.exec(statement.order_by(SourceDocument.uploaded_at.desc())).all()
        return [self._summary_for_source(source.source_id, SourceSummary) for source in sources]

    def search(
        self,
        query: str,
        topic_hint: str | None = None,
        keyword_top_k: int | None = None,
        semantic_top_k: int | None = None,
    ):
        """Run approved-only keyword and semantic retrieval.

        Returns:
            tuple: `(keyword_results, semantic_results, semantic_error)`.
        """

        clean_query = normalize_query(query)
        keyword_results = KeywordRetriever(self.session).search(
            clean_query,
            keyword_top_k or self.settings.keyword_top_k,
            topic_hint,
        )
        semantic_retriever = SemanticRetriever(
            self.session,
            self.settings,
            VectorIndexer(self.settings, self.llm_provider),
        )
        semantic_results, semantic_error = safe_semantic_search(
            semantic_retriever,
            clean_query,
            semantic_top_k or self.settings.semantic_top_k,
            topic_hint,
        )
        return keyword_results, semantic_results, semantic_error

    def add_neighbor_chunks(
        self,
        ranked_chunks: list[RetrievedChunk],
        neighbor_window: int = 1,
    ) -> list[RetrievedChunk]:
        """Add previous and next approved chunks around retrieved chunks.

        Parameters:
            ranked_chunks: RRF-ranked chunks with shape `[chunk_count]`.
            neighbor_window: number of chunks before and after each chunk. V1
                uses `1` to match the outline.
        Returns:
            list[RetrievedChunk]: chunks in context order, deduplicated by
            `chunk_id`, including each retrieved chunk plus up to one previous
            and one next approved chunk from the same source.
        """

        expanded: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        for retrieved in ranked_chunks:
            database_chunk = self.session.get(KnowledgeChunk, retrieved.chunk_id)
            if database_chunk is None:
                continue
            neighbor_indexes = range(
                database_chunk.chunk_index - neighbor_window,
                database_chunk.chunk_index + neighbor_window + 1,
            )
            for chunk_index in neighbor_indexes:
                neighbor = self.session.exec(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.source_id == database_chunk.source_id,
                        KnowledgeChunk.chunk_index == chunk_index,
                        KnowledgeChunk.approval_status == "approved",
                    )
                ).first()
                if neighbor is None or neighbor.chunk_id in seen_chunk_ids:
                    continue
                source = self.session.get(SourceDocument, neighbor.source_id)
                if source is None or source.approval_status != "approved":
                    continue
                seen_chunk_ids.add(neighbor.chunk_id)
                expanded.append(
                    RetrievedChunk(
                        source_id=neighbor.source_id,
                        chunk_id=neighbor.chunk_id,
                        title=source.title,
                        text=neighbor.text,
                        page_number=neighbor.page_number,
                        section_heading=neighbor.section_heading,
                        topic_tags=json.loads(neighbor.topic_tags_json or "[]"),
                        token_count=neighbor.token_count,
                        score=retrieved.score,
                        rank=retrieved.rank,
                        retrieval_method=(
                            retrieved.retrieval_method
                            if neighbor.chunk_id == retrieved.chunk_id
                            else "neighbor"
                        ),
                    )
                )
        return expanded

    def _get_source(self, source_id: str) -> SourceDocument:
        source = self.session.get(SourceDocument, source_id)
        if source is None:
            raise NotFoundError(f"Source not found: {source_id}")
        return source

    def _summary_for_source(self, source_id: str, result_type):
        """Build an API summary from current database state."""

        source = self._get_source(source_id)
        chunk_count = self.session.exec(
            select(func.count(KnowledgeChunk.chunk_id)).where(KnowledgeChunk.source_id == source_id)
        ).one()
        return result_type(
            source_id=source.source_id,
            title=source.title,
            source_type=source.source_type,
            approval_status=source.approval_status,
            parser_status=source.parser_status,
            parser_error=source.parser_error,
            topic_tags=json.loads(source.topic_tags_json or "[]"),
            chunk_count=chunk_count,
        )


def _title_from_file_name(source_path: Path) -> str:
    """Input: source file path. Output: readable title from the file name."""

    return source_path.stem.strip() or source_path.name


def _tags_from_corpus_path(source_path: Path, corpus_root: Path) -> list[str]:
    """Infer topic tags from folder names under `RAG_corpus`.

    Parameters:
        source_path: source file inside the corpus.
        corpus_root: root folder being bulk-uploaded.
    Returns:
        list[str]: folder-name tags with shape `[tag_count]`.
    """

    try:
        relative_parent = source_path.parent.resolve().relative_to(corpus_root.resolve())
    except ValueError:
        return []
    return [part for part in relative_parent.parts if part and part != "."]


def _resolve_corpus_relative_path(corpus_root: Path, relative_path: str) -> Path:
    """Resolve a user-provided corpus relative path without leaving the root."""

    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("RAG corpus paths must be relative paths inside RAG_corpus.")
    resolved = (corpus_root / candidate).resolve()
    if corpus_root not in resolved.parents:
        raise ValueError("RAG corpus path escapes the configured corpus folder.")
    return resolved
