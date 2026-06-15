"""Q&A integration with the User Profile and Memory block."""

from pathlib import Path

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.llm_provider import QueryPlanResult
from app.knowledge.schemas import RetrievedChunk
from app.memory.service import MemoryService
from app.qa.service import QAService
from app.storage.database import create_app_engine, init_db
from app.storage.models import ConversationMessage


class FakeKnowledgeService:
    """Small approved-evidence source for QA tests."""

    def search(self, query: str, topic_hint: str | None = None, keyword_top_k=None, semantic_top_k=None):
        chunk = RetrievedChunk(
            source_id="src_memorytest",
            chunk_id="chk_memorytest",
            title="Protein Note",
            text="Protein supports muscle repair after resistance training.",
            page_number=None,
            section_heading=None,
            topic_tags=["nutrition"],
            token_count=8,
            score=1.0,
            rank=1,
            retrieval_method="keyword",
        )
        return [chunk], [], None

    def add_neighbor_chunks(self, ranked_chunks, neighbor_window: int = 1):
        return ranked_chunks


class FakeProvider:
    """Fake planner and answer provider that records memory context."""

    def __init__(self):
        self.seen_memory_context = None
        self.seen_memory_contexts = []

    def generate_query_plan(self, request):
        return QueryPlanResult(
            clean_question=request.question,
            clarified_question=request.question,
            search_query=request.question,
            query_terms=[request.question],
            notes=[],
        )

    def generate_answer(self, request):
        self.seen_memory_context = request.memory_context
        self.seen_memory_contexts.append(request.memory_context)
        return (
            "Short answer:\n"
            "Use the approved protein evidence as the basis for recovery guidance.\n\n"
            "Practical guidance:\n"
            "Tie it to the user's goal when helpful.\n\n"
            "Evidence used:\n"
            "- [src_memorytest / chk_memorytest] Protein Note, section unavailable\n\n"
            "Safety note:\n"
            "No special safety concern was identified from the evidence."
        )


def test_qa_include_memory_passes_context_and_logs_messages(tmp_path: Path):
    """Q&A can receive memory context without coupling to memory internals."""

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
    provider = FakeProvider()
    memory = MemoryService(session=session, settings=settings, llm_provider=provider)
    memory.create_user("usr_test")
    conversation = memory.create_conversation("usr_test", title="QA with memory")
    memory.update_profile("usr_test", {"primary_training_goal": "build muscle"})

    qa = QAService(
        session=session,
        settings=settings,
        knowledge_service=FakeKnowledgeService(),
        llm_provider=provider,
        memory_service=memory,
    )
    response = qa.ask(
        "How should I think about protein after lifting?",
        topic_hint="nutrition",
        debug=True,
        user_id="usr_test",
        conversation_id=conversation.conversation_id,
        include_memory=True,
        memory_debug=True,
    )
    trace = qa.get_trace(response.trace_id)
    messages = session.exec(
        select(ConversationMessage).where(
            ConversationMessage.conversation_id == conversation.conversation_id
        )
    ).all()

    assert "primary_training_goal: build muscle" in provider.seen_memory_context
    assert response.debug["memory_context"]["plan_context_level"] == "none"
    assert response.debug["memory_debug_ids"]
    assert trace.user_id == "usr_test"
    assert trace.conversation_id == conversation.conversation_id
    assert trace.memory_debug_ids == response.debug["memory_debug_ids"]
    assert [message.role for message in messages] == ["user", "assistant"]
    session.close()


def test_qa_auto_creates_supplied_memory_conversation_and_reuses_working_memory(tmp_path: Path):
    """Q&A should create a typed conversation ID and reuse prior turns on follow-up."""

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
    provider = FakeProvider()
    memory = MemoryService(session=session, settings=settings, llm_provider=provider)
    memory.create_user("usr_test")
    qa = QAService(
        session=session,
        settings=settings,
        knowledge_service=FakeKnowledgeService(),
        llm_provider=provider,
        memory_service=memory,
    )

    first = qa.ask(
        "What should I eat after lifting?",
        topic_hint="nutrition",
        debug=True,
        user_id="usr_test",
        conversation_id="250pt",
        include_memory=True,
        memory_debug=True,
    )
    second = qa.ask(
        "Can you connect that to my recovery?",
        topic_hint="nutrition",
        debug=True,
        user_id="usr_test",
        conversation_id="250pt",
        include_memory=True,
        memory_debug=True,
    )
    detail = memory.get_conversation_detail("250pt")

    assert first.debug["errors"] == []
    assert second.debug["errors"] == []
    assert [item.conversation_id for item in memory.list_conversations("usr_test")] == ["250pt"]
    assert [message.role for message in detail.messages] == ["user", "assistant", "user", "assistant"]
    assert "What should I eat after lifting?" in provider.seen_memory_contexts[-1]
    assert "Short answer:" in provider.seen_memory_contexts[-1]
    assert "Can you connect that to my recovery?" in provider.seen_memory_contexts[-1]
    session.close()
