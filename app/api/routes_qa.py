"""FastAPI routes for the evidence-based Q&A workflow."""

from fastapi import APIRouter, Depends, HTTPException

from app.core.errors import NotFoundError
from app.qa.schemas import AnswerTraceOut, AskQuestionRequest, AskQuestionResponse
from app.qa.service import QAService
from app.api.dependencies import qa_service_dependency

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.post("/ask", response_model=AskQuestionResponse)
def ask_question(
    request: AskQuestionRequest,
    qa_service: QAService = Depends(qa_service_dependency),
) -> AskQuestionResponse:
    """Answer one evidence-grounded question."""

    return qa_service.ask(
        question=request.question,
        topic_hint=request.topic_hint,
        debug=request.debug,
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        include_memory=request.include_memory,
        memory_debug=request.memory_debug,
    )


@router.get("/traces/{trace_id}", response_model=AnswerTraceOut)
def get_answer_trace(
    trace_id: str,
    qa_service: QAService = Depends(qa_service_dependency),
) -> AnswerTraceOut:
    """Return a replayable trace for one previous answer."""

    try:
        return qa_service.get_trace(trace_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
