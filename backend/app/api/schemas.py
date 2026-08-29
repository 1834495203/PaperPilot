from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.entities import AgentEvent, Conversation, ConversationMetrics, Message
from app.domain.enums import MessageRole
from app.domain.types import JsonValue


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(default="New research", min_length=1, max_length=200)


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    content: str = Field(min_length=1, max_length=10_000)


class ConversationResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, conversation: Conversation) -> "ConversationResponse":
        return cls(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    sequence: int
    created_at: datetime
    metadata: dict[str, JsonValue]

    @classmethod
    def from_domain(cls, message: Message) -> "MessageResponse":
        return cls(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            sequence=message.sequence,
            created_at=message.created_at,
            metadata=message.metadata,
        )


class ConversationMetricsResponse(BaseModel):
    conversation_id: UUID
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int
    total_duration_ms: int
    run_count: int
    updated_at: datetime | None

    @classmethod
    def from_domain(cls, metrics: ConversationMetrics) -> "ConversationMetricsResponse":
        return cls(
            conversation_id=metrics.conversation_id,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            llm_calls=metrics.llm_calls,
            tool_calls=metrics.tool_calls,
            total_duration_ms=metrics.total_duration_ms,
            run_count=metrics.run_count,
            updated_at=metrics.updated_at,
        )


class EventResponse(BaseModel):
    id: UUID
    run_id: UUID
    conversation_id: UUID
    sequence: int
    type: str
    timestamp: datetime
    payload: dict[str, JsonValue]

    @classmethod
    def from_domain(cls, event: AgentEvent) -> "EventResponse":
        return cls(
            id=event.id,
            run_id=event.run_id,
            conversation_id=event.conversation_id,
            sequence=event.sequence,
            type=event.type.value,
            timestamp=event.timestamp,
            payload=event.payload,
        )
