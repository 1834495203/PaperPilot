import json
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service, get_paper_library
from app.api.schemas import (
    AgentRunResponse,
    ConversationMetricsResponse,
    ConversationResponse,
    CreateConversationRequest,
    EventResponse,
    MessageResponse,
    SendMessageRequest,
)
from app.application.chat_service import (
    ChatService,
    ConversationBusyError,
    ConversationNotFoundError,
    MessageNotFoundError,
)
from app.application.paper_library import PaperLibraryService
from app.domain.entities import AgentEvent

router = APIRouter(prefix="/conversations", tags=["conversations"])

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _format_event(event: AgentEvent) -> str:
    response = EventResponse.from_domain(event)
    data = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
    return f"id: {event.sequence}\nevent: {event.type.value}\ndata: {data}\n\n"


def _failure_frame(message: str) -> str:
    return f"event: run.failed\ndata: {json.dumps({'error': message})}\n\n"


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: CreateConversationRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ConversationResponse:
    conversation = await service.create_conversation(request.title)
    return ConversationResponse.from_domain(conversation)


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> list[ConversationResponse]:
    conversations = await service.list_conversations()
    return [ConversationResponse.from_domain(item) for item in conversations]


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> None:
    try:
        await service.delete_conversation(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from error
    except ConversationBusyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation has an active run",
        ) from error


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def list_messages(
    conversation_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
    include_superseded: Annotated[bool, Query()] = False,
) -> list[MessageResponse]:
    """Thread messages; regenerated versions stay available behind a flag."""

    try:
        messages = await service.get_messages(
            conversation_id,
            include_superseded=include_superseded,
        )
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from error
    return [MessageResponse.from_domain(item) for item in messages]


@router.get("/{conversation_id}/metrics", response_model=ConversationMetricsResponse)
async def get_conversation_metrics(
    conversation_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ConversationMetricsResponse:
    try:
        metrics = await service.get_conversation_metrics(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from error
    return ConversationMetricsResponse.from_domain(metrics)


@router.get("/{conversation_id}/runs", response_model=list[AgentRunResponse])
async def list_runs(
    conversation_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> list[AgentRunResponse]:
    try:
        runs = await service.get_runs(conversation_id)
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from error
    return [AgentRunResponse.from_domain(run) for run in runs]


@router.get("/{conversation_id}/events", response_model=list[EventResponse])
async def list_events(
    conversation_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[EventResponse]:
    try:
        events = await service.get_events(conversation_id, limit)
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from error
    return [EventResponse.from_domain(event) for event in events]


@router.post(
    "/{conversation_id}/runs/{run_id}/cancel",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def cancel_run(
    conversation_id: UUID,
    run_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> None:
    try:
        cancelled = await service.cancel_run(conversation_id, run_id)
    except ConversationNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        ) from error
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Run is not active",
        )


@router.get("/{conversation_id}/runs/{run_id}/stream")
async def resume_run_stream(
    conversation_id: UUID,
    run_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
    after: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    """Re-attach to a run after a dropped connection.

    Replays buffered events after ``after`` and keeps following the run while it is
    active. A run that is no longer in memory is replayed from persisted events, so
    the client can restore its trace and then reload the message list.
    """

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in service.resume_run(
                conversation_id,
                run_id,
                after_sequence=after,
            ):
                yield _format_event(event)
        except ConversationNotFoundError:
            yield _failure_frame("Conversation not found")
        except MessageNotFoundError:
            yield _failure_frame("Run not found")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/{conversation_id}/messages/{message_id}/regenerate")
async def regenerate_message(
    conversation_id: UUID,
    message_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
    paper_library: Annotated[PaperLibraryService, Depends(get_paper_library)],
) -> StreamingResponse:
    local_corpus_available = bool(await paper_library.list_papers())

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in service.regenerate_message(
                conversation_id,
                message_id,
                local_corpus_available=local_corpus_available,
            ):
                yield _format_event(event)
        except ConversationNotFoundError:
            yield _failure_frame("Conversation not found")
        except MessageNotFoundError:
            yield _failure_frame("Message not found")
        except ConversationBusyError as error:
            yield _failure_frame(str(error))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: UUID,
    request: SendMessageRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    paper_library: Annotated[PaperLibraryService, Depends(get_paper_library)],
) -> StreamingResponse:
    local_corpus_available = bool(await paper_library.list_papers())

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in service.stream_message(
                conversation_id,
                request.content,
                local_corpus_available=local_corpus_available,
            ):
                yield _format_event(event)
        except ConversationNotFoundError:
            yield _failure_frame("Conversation not found")
        except ConversationBusyError as error:
            yield _failure_frame(str(error))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
