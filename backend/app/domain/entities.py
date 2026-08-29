from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.enums import EventType, MessageRole, RunStatus
from app.domain.types import JsonValue


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Conversation:
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Message:
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    sequence: int
    created_at: datetime
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    id: UUID
    run_id: UUID
    message_id: UUID | None
    tool_call_id: str
    tool_name: str
    arguments: dict[str, JsonValue]
    result: JsonValue | None
    error: str | None
    duration_ms: int | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RunMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class ConversationMetrics:
    conversation_id: UUID
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    total_duration_ms: int = 0
    run_count: int = 0
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AgentRun:
    id: UUID
    conversation_id: UUID
    status: RunStatus
    metrics: RunMetrics
    error: str | None
    started_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class AgentEvent:
    id: UUID
    run_id: UUID
    conversation_id: UUID
    sequence: int
    type: EventType
    timestamp: datetime
    payload: dict[str, JsonValue]

    @classmethod
    def create(
        cls,
        *,
        run_id: UUID,
        conversation_id: UUID,
        sequence: int,
        event_type: EventType,
        payload: dict[str, JsonValue],
    ) -> "AgentEvent":
        return cls(
            id=uuid4(),
            run_id=run_id,
            conversation_id=conversation_id,
            sequence=sequence,
            type=event_type,
            timestamp=utc_now(),
            payload=payload,
        )
