"""Indexing behavior for approved knowledge chunks."""

from app.core.config import Settings
from app.knowledge.indexing import VectorIndexer
from app.storage.models import KnowledgeChunk, SourceDocument


class FakeEmbeddingProvider:
    """Fake provider that records each embedding request shape."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def embed_texts(self, texts):
        """Input: `[text_count]`. Output: `[text_count, 3]` fake vectors."""

        self.batch_sizes.append(len(texts))
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]


class FakeCollection:
    """Small in-memory stand-in for a Chroma collection."""

    def __init__(self):
        self.upsert_sizes: list[int] = []
        self.deleted_sources: list[str] = []

    def upsert(self, ids, documents, embeddings, metadatas):
        """Record the number of chunks upserted in one batch."""

        self.upsert_sizes.append(len(ids))
        assert len(ids) == len(documents) == len(embeddings) == len(metadatas)

    def delete(self, where):
        """Record source cleanup calls."""

        self.deleted_sources.append(where["source_id"])


def test_vector_indexer_batches_large_sources(monkeypatch):
    """Large sources should not be sent to the embedding provider in one request."""

    settings = Settings(_env_file=None, embedding_batch_size=2)
    provider = FakeEmbeddingProvider()
    collection = FakeCollection()
    indexer = VectorIndexer(settings=settings, embedding_provider=provider)
    monkeypatch.setattr(indexer, "_get_collection", lambda: collection)
    source = SourceDocument(
        source_id="src_test",
        title="Large PDF",
        source_type="pdf",
        file_path="large.pdf",
        approval_status="approved",
    )
    chunks = [
        KnowledgeChunk(
            chunk_id=f"chk_{index}",
            source_id=source.source_id,
            chunk_index=index,
            text=f"chunk text {index}",
            approval_status="approved",
        )
        for index in range(5)
    ]

    count = indexer.index_chunks(source, chunks)

    assert count == 5
    assert provider.batch_sizes == [2, 2, 1]
    assert collection.upsert_sizes == [2, 2, 1]
    assert collection.deleted_sources == ["src_test"]


def test_vector_indexer_caps_dashscope_batches_at_ten(monkeypatch):
    """A DashScope config of 16 should still send provider-safe batches of 10."""

    settings = Settings(_env_file=None, embed_model_type="dashscope", embedding_batch_size=16)
    provider = FakeEmbeddingProvider()
    collection = FakeCollection()
    indexer = VectorIndexer(settings=settings, embedding_provider=provider)
    monkeypatch.setattr(indexer, "_get_collection", lambda: collection)
    source = SourceDocument(
        source_id="src_test",
        title="Large PDF",
        source_type="pdf",
        file_path="large.pdf",
        approval_status="approved",
    )
    chunks = [
        KnowledgeChunk(
            chunk_id=f"chk_{index}",
            source_id=source.source_id,
            chunk_index=index,
            text=f"chunk text {index}",
            approval_status="approved",
        )
        for index in range(21)
    ]

    count = indexer.index_chunks(source, chunks)

    assert count == 21
    assert provider.batch_sizes == [10, 10, 1]
