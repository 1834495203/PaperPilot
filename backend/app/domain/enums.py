from enum import StrEnum


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(StrEnum):
    RUN_STARTED = "run.started"
    STAGE_STARTED = "stage.started"
    DECISION_RECORDED = "decision.recorded"
    TOKEN = "message.token"
    MESSAGE_COMPLETED = "message.completed"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    METRICS_UPDATED = "metrics.updated"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


PERSISTED_AGENT_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.RUN_STARTED,
        EventType.STAGE_STARTED,
        EventType.DECISION_RECORDED,
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.TOOL_FAILED,
        EventType.METRICS_UPDATED,
        EventType.RUN_COMPLETED,
        EventType.RUN_FAILED,
    }
)
