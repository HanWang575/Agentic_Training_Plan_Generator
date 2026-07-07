"""FastAPI routes for the User Profile and Memory block."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import memory_service_dependency
from app.core.errors import NotFoundError
from app.memory.schemas import (
    CloseConversationRequest,
    ConversationCloseResponse,
    ConversationCreateRequest,
    ConversationDetailOut,
    ConversationMessageOut,
    ConversationOut,
    EpisodeSearchResponse,
    ExpiringRecentStatusResponse,
    KeyProposalOut,
    KeyProposalReviewRequest,
    MemoryContextOut,
    MemoryDeleteResponse,
    MemoryItemApproveRequest,
    MemoryItemExpireRequest,
    MemoryItemOut,
    MemoryItemRejectRequest,
    MemoryItemReviewResponse,
    PlanProgressOut,
    PlanProgressPatchRequest,
    PlanActivationRequest,
    PlanActivationResponse,
    ProfilePatchRequest,
    ProfilePatchResponse,
    ProfileResponse,
    RecentStatusExpireRequest,
    RecentStatusOut,
    RecentStatusPatchRequest,
    RecentStatusPatchResponse,
    RecentStatusValidUntilRequest,
    TrainingLogImportRequest,
    TrainingLogImportResponse,
    TrainingLogXunjiSyncRequest,
    TrainingLogXunjiSyncResponse,
    TrainingSessionListResponse,
    UserCreateRequest,
    UserOut,
    MessageAppendRequest,
)
from app.memory.service import MemoryService

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.get("/users", response_model=list[UserOut])
def list_users(
    service: MemoryService = Depends(memory_service_dependency),
) -> list[UserOut]:
    """List local users."""

    return service.list_users()


@router.post("/users", response_model=UserOut)
def create_user(
    request: UserCreateRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> UserOut:
    """Create or update a local user."""

    return service.create_user(
        user_id=request.user_id,
        display_name=request.display_name,
        timezone_name=request.timezone,
    )


@router.delete("/users/{user_id}", response_model=MemoryDeleteResponse)
def delete_user(
    user_id: str,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryDeleteResponse:
    """Delete a local user and all user-owned memory rows."""

    try:
        return service.delete_user(user_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    user_id: str | None = None,
    service: MemoryService = Depends(memory_service_dependency),
) -> list[ConversationOut]:
    """List memory-backed conversations."""

    return service.list_conversations(user_id=user_id)


@router.post("/conversations", response_model=ConversationOut)
def create_conversation(
    request: ConversationCreateRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> ConversationOut:
    """Start a memory-backed conversation."""

    try:
        return service.create_conversation(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            title=request.title,
            topic=request.topic,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
def get_conversation_detail(
    conversation_id: str,
    service: MemoryService = Depends(memory_service_dependency),
) -> ConversationDetailOut:
    """Return one conversation and all saved messages."""

    try:
        return service.get_conversation_detail(conversation_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/conversations/{conversation_id}", response_model=MemoryDeleteResponse)
def delete_conversation(
    conversation_id: str,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryDeleteResponse:
    """Delete a conversation and its directly linked memory rows."""

    try:
        return service.delete_conversation(conversation_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/messages", response_model=ConversationMessageOut)
def append_message(
    conversation_id: str,
    request: MessageAppendRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> ConversationMessageOut:
    """Append one message to working memory."""

    try:
        return service.append_message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            source_block=request.source_block,
            qa_trace_id=request.qa_trace_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/conversations/{conversation_id}/context", response_model=MemoryContextOut)
def get_conversation_context(
    conversation_id: str,
    query: str = "",
    task_type: str = "qa",
    memory_debug: bool = False,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryContextOut:
    """Retrieve memory context for one conversation."""

    try:
        return service.get_conversation_context(
            conversation_id=conversation_id,
            query=query,
            task_type=task_type,
            include_debug=memory_debug,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/context", response_model=MemoryContextOut)
def get_memory_context(
    user_id: str,
    query: str = "",
    conversation_id: str | None = None,
    task_type: str = "qa",
    memory_debug: bool = False,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryContextOut:
    """Retrieve memory context for a user, with optional conversation scope."""

    try:
        return service.get_memory_context(
            user_id=user_id,
            query=query,
            conversation_id=conversation_id,
            task_type=task_type,
            include_debug=memory_debug,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/conversations/{conversation_id}/close", response_model=ConversationCloseResponse)
def close_conversation(
    conversation_id: str,
    request: CloseConversationRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> ConversationCloseResponse:
    """Close a conversation and create an episode."""

    try:
        return service.close_conversation(
            conversation_id=conversation_id,
            closure_reason=request.closure_reason,
            extract_candidates=request.extract_candidates,
            program_context_id=request.program_context_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/episodes/search", response_model=EpisodeSearchResponse)
def search_episodes(
    user_id: str,
    query: str = "",
    top_k: int = 5,
    service: MemoryService = Depends(memory_service_dependency),
) -> EpisodeSearchResponse:
    """Search episodic memories."""

    return service.search_episodes(user_id=user_id, query=query, top_k=top_k)


@router.get("/keys/profile", response_model=list[str])
def list_profile_keys(
    user_id: str | None = None,
    service: MemoryService = Depends(memory_service_dependency),
) -> list[str]:
    """Return approved profile field keys."""

    return service.list_profile_keys(user_id=user_id)


@router.get("/keys/recent-status", response_model=list[str])
def list_recent_status_keys(
    user_id: str | None = None,
    service: MemoryService = Depends(memory_service_dependency),
) -> list[str]:
    """Return approved recent-status keys."""

    return service.list_recent_status_keys(user_id=user_id)


@router.get("/profile/{user_id}", response_model=ProfileResponse)
def get_profile(
    user_id: str,
    service: MemoryService = Depends(memory_service_dependency),
) -> ProfileResponse:
    """Return approved profile fields."""

    return service.list_profile(user_id)


@router.delete("/profile-fields/{profile_field_id}", response_model=MemoryDeleteResponse)
def delete_profile_field(
    profile_field_id: str,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryDeleteResponse:
    """Delete one profile entry by ID."""

    try:
        return service.delete_profile_field(profile_field_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/profile/{user_id}", response_model=ProfilePatchResponse)
def patch_profile(
    user_id: str,
    request: ProfilePatchRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> ProfilePatchResponse:
    """Manually edit approved profile fields."""

    try:
        return service.update_profile(
            user_id=user_id,
            fields=request.fields,
            source=request.source,
            extension_mode=request.extension_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/recent-status/current", response_model=list[RecentStatusOut])
def list_current_recent_status(
    user_id: str,
    include_expired: bool = False,
    service: MemoryService = Depends(memory_service_dependency),
) -> list[RecentStatusOut]:
    """Return current active or audit recent status rows."""

    return service.list_recent_status(user_id=user_id, include_expired=include_expired)


@router.delete("/recent-status/{status_id}", response_model=MemoryDeleteResponse)
def delete_recent_status(
    status_id: str,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryDeleteResponse:
    """Delete one recent-status entry by ID."""

    try:
        return service.delete_recent_status(status_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/recent-status/users/{user_id}", response_model=RecentStatusPatchResponse)
def patch_recent_status(
    user_id: str,
    request: RecentStatusPatchRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> RecentStatusPatchResponse:
    """Manually edit recent status values."""

    try:
        return service.update_recent_status(
            user_id=user_id,
            statuses=request.statuses,
            observed_at=request.observed_at,
            source=request.source,
            extension_mode=request.extension_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/items", response_model=list[MemoryItemOut])
def list_memory_items(
    user_id: str | None = None,
    status: str | None = None,
    category: str | None = None,
    service: MemoryService = Depends(memory_service_dependency),
) -> list[MemoryItemOut]:
    """List candidate, approved, rejected, or expired memory items."""

    return service.list_memory_items(user_id=user_id, status=status, category=category)


@router.post("/items/{memory_id}/approve", response_model=MemoryItemReviewResponse)
def approve_memory_item(
    memory_id: str,
    request: MemoryItemApproveRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryItemReviewResponse:
    """Approve or edit a candidate memory item."""

    try:
        return service.approve_memory_item(memory_id, request)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/items/{memory_id}/reject", response_model=MemoryItemReviewResponse)
def reject_memory_item(
    memory_id: str,
    request: MemoryItemRejectRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryItemReviewResponse:
    """Reject a candidate memory item."""

    try:
        return service.reject_memory_item(memory_id, review_note=request.review_note)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/items/{memory_id}/expire", response_model=MemoryItemReviewResponse)
def expire_memory_item(
    memory_id: str,
    request: MemoryItemExpireRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> MemoryItemReviewResponse:
    """Expire a memory item and any linked recent status."""

    try:
        return service.expire_memory_item(memory_id, reason=request.reason)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/key-proposals", response_model=list[KeyProposalOut])
def list_key_proposals(
    user_id: str | None = None,
    review_status: str | None = None,
    service: MemoryService = Depends(memory_service_dependency),
) -> list[KeyProposalOut]:
    """List memory key proposals."""

    return service.list_key_proposals(user_id=user_id, review_status=review_status)


@router.post("/key-proposals/{proposal_id}/approve", response_model=KeyProposalOut)
def approve_key_proposal(
    proposal_id: str,
    request: KeyProposalReviewRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> KeyProposalOut:
    """Approve a proposed memory key."""

    try:
        return service.approve_key_proposal(
            proposal_id,
            proposed_key=request.proposed_key,
            target_type=request.target_type,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/key-proposals/{proposal_id}/reject", response_model=KeyProposalOut)
def reject_key_proposal(
    proposal_id: str,
    service: MemoryService = Depends(memory_service_dependency),
) -> KeyProposalOut:
    """Reject a proposed memory key."""

    try:
        return service.reject_key_proposal(proposal_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/recent-status/expiring", response_model=ExpiringRecentStatusResponse)
def list_expiring_recent_status(
    user_id: str,
    within_hours: int = 2,
    service: MemoryService = Depends(memory_service_dependency),
) -> ExpiringRecentStatusResponse:
    """Return recent statuses close to expiration."""

    return service.list_expiring_recent_status(user_id=user_id, within_hours=within_hours)


@router.patch("/recent-status/{status_id}/valid-until", response_model=RecentStatusOut)
def update_recent_status_valid_until(
    status_id: str,
    request: RecentStatusValidUntilRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> RecentStatusOut:
    """Edit one recent status valid_until timestamp."""

    try:
        return service.update_recent_status_valid_until(status_id, request.valid_until)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/recent-status/{status_id}/expire", response_model=RecentStatusOut)
def expire_recent_status(
    status_id: str,
    request: RecentStatusExpireRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> RecentStatusOut:
    """Expire one recent status while keeping it for audit."""

    try:
        return service.expire_recent_status(status_id, reason=request.reason)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/plans/active", response_model=PlanProgressOut | None)
def get_active_plan(
    user_id: str,
    service: MemoryService = Depends(memory_service_dependency),
) -> PlanProgressOut | None:
    """Return the current active plan progress, if present."""

    return service.get_active_plan(user_id)


@router.post("/plans/activate", response_model=PlanActivationResponse)
def activate_plan(
    request: PlanActivationRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> PlanActivationResponse:
    """Atomically replace the user's active plan."""

    return service.activate_plan(request)


@router.patch("/plans/{plan_id}/progress", response_model=PlanProgressOut)
def patch_plan_progress(
    plan_id: str,
    request: PlanProgressPatchRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> PlanProgressOut:
    """Create or update plan progress and append an event."""

    try:
        return service.patch_plan_progress(plan_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/imports/xunji", response_model=TrainingLogImportResponse)
def import_xunji(
    request: TrainingLogImportRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> TrainingLogImportResponse:
    """Import Xunji training logs idempotently."""

    return service.import_xunji(request)


@router.post("/imports/xunji/sync", response_model=TrainingLogXunjiSyncResponse)
def sync_xunji_training_logs(
    request: TrainingLogXunjiSyncRequest,
    service: MemoryService = Depends(memory_service_dependency),
) -> TrainingLogXunjiSyncResponse:
    """Read Xunji logs from the Open API and import normalized sessions."""

    try:
        return service.sync_xunji_training_logs(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/training-log/sessions", response_model=TrainingSessionListResponse)
def list_training_sessions(
    user_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    sport_type: str | None = None,
    limit: int = 200,
    service: MemoryService = Depends(memory_service_dependency),
) -> TrainingSessionListResponse:
    """List normalized training sessions by date range and sport type."""

    return service.list_training_sessions(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        sport_type=sport_type,
        limit=limit,
    )

