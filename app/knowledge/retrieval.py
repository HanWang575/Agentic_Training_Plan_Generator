"""Approved-only keyword and semantic retrieval."""

import json
import re

from sqlalchemy import text
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.knowledge.indexing import VectorIndexer
from app.knowledge.schemas import RetrievedChunk
from app.storage.models import KnowledgeChunk, SourceDocument


def normalize_query(query: str) -> str:
    """Input: raw user query. Output: whitespace-normalized query string."""

    return re.sub(r"\s+", " ", query).strip()


def expand_query_terms(query: str) -> list[str]:
    """Generate a few transparent search terms for v1 keyword retrieval."""

    lowered = query.lower()
    terms = [query]
    synonym_map = {
        "protein": ["amino acids", "muscle protein synthesis"],
        "hyrox": ["hybrid training", "race rulebook"],
        "powerlifting": ["squat bench deadlift", "1rm"],
        "bodybuilding": ["hypertrophy", "muscle growth"],
        "running": ["endurance", "vo2 max"],
    }
    for trigger, synonyms in synonym_map.items():
        if trigger in lowered:
            terms.extend(synonyms)
    return terms[:8]


class KeywordRetriever:
    """SQLite FTS5 retriever that only returns approved chunks."""

    def __init__(self, session: Session):
        self.session = session

    def search(self, query: str, top_k: int, topic_hint: str | None = None) -> list[RetrievedChunk]:
        """Search the FTS5 index.

        Parameters:
            query: normalized query string.
            top_k: max result count.
            topic_hint: optional soft filter; results are still ranked by bm25.
        Returns:
            list[RetrievedChunk]: shape `[0..top_k]`.
        """

        match_query = _to_fts_query(query)
        if not match_query:
            return []

        rows = self.session.exec(
            text(
                """
                SELECT
                    f.chunk_id,
                    f.source_id,
                    bm25(knowledge_chunks_fts) AS bm25_score
                FROM knowledge_chunks_fts AS f
                WHERE knowledge_chunks_fts MATCH :match_query
                ORDER BY bm25_score
                LIMIT :top_k
                """
            ),
            params={"match_query": match_query, "top_k": top_k},
        ).all()

        results: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            chunk = self.session.get(KnowledgeChunk, row.chunk_id)
            if chunk is None or chunk.approval_status != "approved":
                continue
            source = self.session.get(SourceDocument, chunk.source_id)
            if source is None or source.approval_status != "approved":
                continue
            if topic_hint and topic_hint not in json.loads(chunk.topic_tags_json or "[]"):
                # Topic hints are soft: keep the row, but do not boost it.
                score = 1.0 / (rank + 3)
            else:
                score = 1.0 / rank
            results.append(_to_retrieved_chunk(chunk, source, score, rank, "keyword"))
        return results


class SemanticRetriever:
    """Chroma retriever using the configured embedding provider."""

    def __init__(self, session: Session, settings: Settings, vector_indexer: VectorIndexer):
        self.session = session
        self.settings = settings
        self.vector_indexer = vector_indexer

    def search(self, query: str, top_k: int, topic_hint: str | None = None) -> list[RetrievedChunk]:
        """Search the vector database.

        Returns:
            list[RetrievedChunk]: semantic results. Raises ConfigurationError if
            embeddings or Chroma are not configured.
        """

        raw = self.vector_indexer.query(query, top_k)
        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        results: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(ids, start=1):
            chunk = self.session.get(KnowledgeChunk, chunk_id)
            if chunk is None or chunk.approval_status != "approved":
                continue
            source = self.session.get(SourceDocument, chunk.source_id)
            if source is None or source.approval_status != "approved":
                continue
            distance = distances[rank - 1] if rank - 1 < len(distances) else 1.0
            score = max(0.0, 1.0 - float(distance))
            if topic_hint and topic_hint not in json.loads(chunk.topic_tags_json or "[]"):
                score *= 0.8
            results.append(_to_retrieved_chunk(chunk, source, score, rank, "semantic"))
        return results


def _to_fts_query(query: str) -> str:
    """Convert free text into a conservative OR query for FTS5."""

    words = re.findall(r"[A-Za-z0-9_]+", query)
    return " OR ".join(words[:16])


def _to_retrieved_chunk(
    chunk: KnowledgeChunk,
    source: SourceDocument,
    score: float,
    rank: int,
    method: str,
) -> RetrievedChunk:
    """Map SQLModel rows into the retrieval dataclass used by QA nodes."""

    return RetrievedChunk(
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        title=source.title,
        text=chunk.text,
        page_number=chunk.page_number,
        section_heading=chunk.section_heading,
        content_type=chunk.content_type,
        content_metadata=json.loads(chunk.content_metadata_json or "{}"),
        topic_tags=json.loads(chunk.topic_tags_json or "[]"),
        token_count=chunk.token_count,
        score=score,
        rank=rank,
        retrieval_method=method,
    )


def safe_semantic_search(
    semantic_retriever: SemanticRetriever,
    query: str,
    top_k: int,
    topic_hint: str | None,
) -> tuple[list[RetrievedChunk], str | None]:
    """Run semantic retrieval and return an error string instead of crashing."""

    try:
        return semantic_retriever.search(query, top_k, topic_hint), None
    except ConfigurationError as exc:
        return [], str(exc)
    except Exception as exc:
        return [], f"Semantic retrieval failed: {exc}"
