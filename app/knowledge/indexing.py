"""Keyword and vector indexing helpers for approved knowledge chunks."""

import json
from collections.abc import Sequence

from sqlalchemy import text
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.llm_provider import OpenAIProvider
from app.storage.models import KnowledgeChunk, SourceDocument


class KeywordIndexer:
    """Maintain the SQLite FTS5 keyword index."""

    def __init__(self, session: Session):
        self.session = session

    def index_source(self, source_id: str) -> int:
        """Index all approved chunks for one source.

        Parameters:
            source_id: source document ID with shape `src_*`.
        Returns:
            int: number of chunks written to the FTS table.
        """

        self.remove_source(source_id)
        source = self.session.get(SourceDocument, source_id)
        if source is None:
            return 0
        chunks = self.session.exec(
            select(KnowledgeChunk).where(
                KnowledgeChunk.source_id == source_id,
                KnowledgeChunk.approval_status == "approved",
            )
        ).all()
        for chunk in chunks:
            self.session.exec(
                text(
                    """
                    INSERT INTO knowledge_chunks_fts
                        (chunk_id, source_id, title, text, topic_tags)
                    VALUES (:chunk_id, :source_id, :title, :text, :topic_tags)
                    """
                ),
                params={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "title": source.title,
                    "text": chunk.text,
                    "topic_tags": " ".join(json.loads(chunk.topic_tags_json or "[]")),
                },
            )
        self.session.commit()
        return len(chunks)

    def remove_source(self, source_id: str) -> None:
        """Remove all FTS rows for a source before re-indexing or rejection."""

        self.session.exec(
            text("DELETE FROM knowledge_chunks_fts WHERE source_id = :source_id"),
            params={"source_id": source_id},
        )
        self.session.commit()


class VectorIndexer:
    """Maintain a persistent Chroma collection for approved chunks."""

    def __init__(self, settings: Settings, embedding_provider: OpenAIProvider):
        self.settings = settings
        self.embedding_provider = embedding_provider
        self._collection = None

    def _get_collection(self):
        """Return the configured Chroma collection, creating it if needed."""

        try:
            import chromadb
        except ImportError as exc:
            raise ConfigurationError("chromadb is not installed.") from exc

        self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        if self._collection is None:
            self._collection = client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index_chunks(self, source: SourceDocument, chunks: list[KnowledgeChunk]) -> int:
        """Embed and upsert approved chunks into Chroma.

        Parameters:
            source: approved source metadata.
            chunks: approved KnowledgeChunk rows with shape `[chunk_count]`.
        Returns:
            int: number of chunks upserted.
        Raises:
            ConfigurationError: if the embedding provider or Chroma is unavailable.
        """

        if not chunks:
            return 0
        collection = self._get_collection()
        self.remove_source(source.source_id)

        indexed_count = 0
        batch_size = self.settings.active_embedding_batch_size
        for batch in _batched(chunks, batch_size):
            embeddings = self.embedding_provider.embed_texts([chunk.text for chunk in batch])
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings,
                metadatas=[
                    {
                        "source_id": chunk.source_id,
                        "title": source.title,
                        "page_number": chunk.page_number or "",
                        "section_heading": chunk.section_heading or "",
                        "topic_tags": chunk.topic_tags_json,
                        "approval_status": chunk.approval_status,
                    }
                    for chunk in batch
                ],
            )
            indexed_count += len(batch)
        return indexed_count

    def remove_source(self, source_id: str) -> None:
        """Remove all vector rows for one source before re-indexing or rollback."""

        collection = self._get_collection()
        collection.delete(where={"source_id": source_id})

    def query(self, query_text: str, top_k: int) -> dict:
        """Embed a query and retrieve Chroma neighbors.

        Parameters:
            query_text: normalized user search string.
            top_k: result count.
        Returns:
            dict: raw Chroma query result.
        """

        collection = self._get_collection()
        query_embedding = self.embedding_provider.embed_texts([query_text])[0]
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where={"approval_status": "approved"},
        )


def _batched(items: Sequence[KnowledgeChunk], batch_size: int) -> list[list[KnowledgeChunk]]:
    """Split chunks into batches.

    Parameters:
        items: chunk sequence with shape `[chunk_count]`.
        batch_size: maximum chunks per embedding request.
    Returns:
        list[list[KnowledgeChunk]] with shape `[batch_count, <=batch_size]`.
    """

    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]
