"""Pydantic and dataclass objects for the knowledge service."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


SUPPORTED_FILE_TYPES = {".pdf", ".epub", ".md", ".txt", ".json"}


@dataclass(slots=True)
class ParsedTextBlock:
    """One text block emitted by a parser.

    Input/source shape:
        page_number and section_heading are optional because not every file type
        has pages or headings.
    """

    text: str
    page_number: int | None = None
    section_heading: str | None = None
    content_type: str = "plain_text"
    content_metadata: dict[str, Any] = field(default_factory=dict)
    quality_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocument:
    """Parser output before chunking."""

    title: str
    source_type: str
    blocks: list[ParsedTextBlock]
    parser_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChunkDraft:
    """Chunker output before database IDs are assigned."""

    text: str
    chunk_index: int
    page_number: int | None
    section_heading: str | None
    token_count: int
    content_type: str = "plain_text"
    content_metadata: dict[str, Any] = field(default_factory=dict)
    quality_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedChunk:
    """Retrieval result used by Q&A graph nodes.

    score is higher-is-better after local normalization.
    rank starts at 1 inside each retrieval method.
    """

    source_id: str
    chunk_id: str
    title: str
    text: str
    page_number: int | None
    section_heading: str | None
    topic_tags: list[str]
    token_count: int
    score: float
    rank: int
    retrieval_method: str
    content_type: str = "plain_text"
    content_metadata: dict[str, Any] = field(default_factory=dict)


class SourceSummary(BaseModel):
    """API output shape for source listing and upload responses."""

    source_id: str
    title: str
    source_type: str
    approval_status: str
    parser_status: str
    parser_progress: float = 0.0
    parser_progress_message: str | None = None
    parser_error: str | None = None
    parser_warnings: list[str] = Field(default_factory=list)
    topic_tags: list[str] = Field(default_factory=list)
    chunk_count: int = 0


class UploadResult(SourceSummary):
    """API output shape for a completed upload parse."""


class BulkUploadResult(BaseModel):
    """API output shape after importing a folder of local source files."""

    uploaded: list[UploadResult] = Field(default_factory=list)
    skipped: list[dict] = Field(default_factory=list)
    uploaded_count: int = 0
    skipped_count: int = 0


class CorpusFileItem(BaseModel):
    """One supported source file discovered under `RAG_corpus`."""

    relative_path: str
    title: str
    topic_tags: list[str] = Field(default_factory=list)
    source_type: str
    size_bytes: int = 0


class CorpusPlanResult(BaseModel):
    """API output shape for previewing files available for corpus upload."""

    files: list[CorpusFileItem] = Field(default_factory=list)
    skipped: list[dict] = Field(default_factory=list)
    file_count: int = 0
    skipped_count: int = 0


class ReviewResult(BaseModel):
    """API output shape after approving or rejecting a source."""

    source_id: str
    approval_status: str
    indexing_status: str
    indexed_chunk_count: int = 0
    message: str | None = None


class BatchReviewRequest(BaseModel):
    """Input shape for approving or rejecting selected/all review sources."""

    action: str
    source_ids: list[str] = Field(default_factory=list)
    review_all: bool = False
    review_note: str | None = None


class BatchReviewResult(BaseModel):
    """Output shape after a batch approve/reject operation."""

    action: str
    results: list[ReviewResult] = Field(default_factory=list)
    failed: list[dict] = Field(default_factory=list)
    processed_count: int = 0
    failed_count: int = 0


class RemoveSourcesRequest(BaseModel):
    """Input shape for removing approved sources from the knowledge base."""

    source_ids: list[str] = Field(default_factory=list)
    remove_all: bool = False


class RemoveSourcesResult(BaseModel):
    """Output shape after removing one, many, or all approved sources."""

    removed_source_ids: list[str] = Field(default_factory=list)
    missing_source_ids: list[str] = Field(default_factory=list)
    removed_count: int = 0
    message: str | None = None


class SearchKnowledgeRequest(BaseModel):
    """Input shape for knowledge search APIs and MCP tools."""

    query: str
    topic_hint: str | None = None
    top_k: int | None = None


class SearchKnowledgeResponse(BaseModel):
    """Output shape for knowledge search APIs and MCP tools."""

    chunks: list[dict]


class ChunkLookupRequest(BaseModel):
    """Input shape for looking up stored chunk text by chunk ID."""

    chunk_ids: list[str] = Field(default_factory=list)


class ChunkDetail(BaseModel):
    """Output shape for one full stored chunk."""

    source_id: str
    chunk_id: str
    title: str
    text: str
    page_number: int | None = None
    section_heading: str | None = None
    content_type: str = "plain_text"
    content_metadata: dict[str, Any] = Field(default_factory=dict)
    topic_tags: list[str] = Field(default_factory=list)
    token_count: int = 0
    approval_status: str
    quality_warnings: list[str] = Field(default_factory=list)


class ChunkLookupResponse(BaseModel):
    """Output shape for full chunk lookup."""

    chunks: list[ChunkDetail] = Field(default_factory=list)
    missing_chunk_ids: list[str] = Field(default_factory=list)
