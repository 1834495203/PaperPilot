import json
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_chat_service
from app.api.schemas import (
    ConversationMetricsResponse,
    ConversationResponse,
    CreateConversationRequest,
    EventResponse,
    MessageResponse,
    SendMessageRequest,
)
from app.application.chat_service import ChatService, ConversationNotFoundError

router = APIRouter(prefix="/conversations", tags=["conversations"])


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


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def list_messages(
    conversation_id: UUID,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> list[MessageResponse]:
    try:
        messages = await service.get_messages(conversation_id)
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


@router.post("/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: UUID,
    request: SendMessageRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in service.stream_message(conversation_id, request.content):
                response = EventResponse.from_domain(event)
                data = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
                yield f"id: {event.sequence}\nevent: {event.type.value}\ndata: {data}\n\n"
        except ConversationNotFoundError:
            error_data = json.dumps({"error": "Conversation not found"})
            yield f"event: run.failed\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
