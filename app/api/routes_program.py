"""FastAPI routes for program generation, review, summaries, and calculators."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import program_service_dependency
from app.core.errors import ConfigurationError, NotFoundError
from app.program.schemas import (
    CalculationResult,
    CalculatorRequest,
    ProgramContextResponse,
    ProgramReviewRequest,
    ProgramReviewResponse,
    ProgramTraceResponse,
    ProgramTurnRequest,
    ProgramTurnResponse,
    ProgramVersionResponse,
    TrainingSummaryNotesResponse,
    WeeklySummaryRefreshRequest,
    XunjiWriteRequest,
    XunjiWriteResponse,
)
from app.program.service import ProgramService


router = APIRouter(prefix="/api", tags=["program-generator"])


def _run(action):
    try:
        return action()
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/programs/conversations/{conversation_id}/turns", response_model=ProgramTurnResponse)
def send_program_turn(
    conversation_id: str,
    request: ProgramTurnRequest,
    service: ProgramService = Depends(program_service_dependency),
) -> ProgramTurnResponse:
    return _run(lambda: service.process_turn(conversation_id, request))


@router.get("/programs/contexts/{context_id}", response_model=ProgramContextResponse)
def get_program_context(
    context_id: str,
    service: ProgramService = Depends(program_service_dependency),
) -> ProgramContextResponse:
    return _run(lambda: service.get_context(context_id))


@router.get("/programs/{program_id}/versions/{version_id}", response_model=ProgramVersionResponse)
def get_program_version(
    program_id: str,
    version_id: str,
    service: ProgramService = Depends(program_service_dependency),
) -> ProgramVersionResponse:
    return _run(lambda: service.get_version(program_id, version_id))


@router.post(
    "/programs/{program_id}/versions/{version_id}/review",
    response_model=ProgramReviewResponse,
)
def review_program(
    program_id: str,
    version_id: str,
    request: ProgramReviewRequest,
    service: ProgramService = Depends(program_service_dependency),
) -> ProgramReviewResponse:
    return _run(lambda: service.review_program(program_id, version_id, request))


@router.post(
    "/programs/{program_id}/versions/{version_id}/xunji-write",
    response_model=XunjiWriteResponse,
)
def write_program_to_xunji(
    program_id: str,
    version_id: str,
    request: XunjiWriteRequest,
    service: ProgramService = Depends(program_service_dependency),
) -> XunjiWriteResponse:
    return _run(lambda: service.write_to_xunji(program_id, version_id, request))


@router.get("/programs/traces/{trace_id}", response_model=ProgramTraceResponse)
def get_program_trace(
    trace_id: str,
    service: ProgramService = Depends(program_service_dependency),
) -> ProgramTraceResponse:
    return _run(lambda: service.get_trace(trace_id))


@router.get("/training/summary-notes", response_model=TrainingSummaryNotesResponse)
def get_training_summary_notes(
    user_id: str,
    limit: int = 4,
    service: ProgramService = Depends(program_service_dependency),
) -> TrainingSummaryNotesResponse:
    return _run(lambda: service.list_weekly_summaries(user_id, limit=limit))


@router.post("/training/summary-notes/refresh", response_model=TrainingSummaryNotesResponse)
def refresh_training_summary_notes(
    request: WeeklySummaryRefreshRequest,
    service: ProgramService = Depends(program_service_dependency),
) -> TrainingSummaryNotesResponse:
    return _run(lambda: service.refresh_weekly_summaries(request.user_id, request.reference_date))


@router.post("/programs/calculators/{tool_name}", response_model=CalculationResult)
def run_program_calculator(
    tool_name: str,
    request: CalculatorRequest,
    service: ProgramService = Depends(program_service_dependency),
) -> CalculationResult:
    return _run(lambda: service.run_calculator(tool_name, request))
