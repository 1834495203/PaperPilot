from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from app.domain.entities import (
    AgentEvent,
    AgentRun,
    Conversation,
    ConversationMetrics,
    Message,
    RunMetrics,
)
from app.domain.enums import MessageRole, RunStatus
from app.domain.papers import ArxivSearchInput, Paper
from app.domain.types import JsonValue


class ConversationStore(ABC):
    @abstractmethod
    async def create_conversation(self, title: str) -> Conversation: ...

    @abstractmethod
    async def list_conversations(self) -> Sequence[Conversation]: ...

    @abstractmethod
    async def get_conversation(self, conversation_id: UUID) -> Conversation | None: ...

    @abstractmethod
    async def list_messages(self, conversation_id: UUID) -> Sequence[Message]: ...

    @abstractmethod
    async def list_events(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> Sequence[AgentEvent]: ...

    @abstractmethod
    async def append_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Message: ...

    @abstractmethod
    async def create_run(self, conversation_id: UUID) -> AgentRun: ...

    @abstractmethod
    async def finish_run(
        self,
        run_id: UUID,
        status: RunStatus,
        metrics: RunMetrics,
        error: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_conversation_metrics(
        self,
        conversation_id: UUID,
    ) -> ConversationMetrics: ...

    @abstractmethod
    async def refresh_conversation_metrics(
        self,
        conversation_id: UUID,
    ) -> ConversationMetrics: ...

    @abstractmethod
    async def append_tool_call(
        self,
        *,
        run_id: UUID,
        message_id: UUID | None,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, JsonValue],
        result: JsonValue | None,
        error: str | None,
        duration_ms: int | None,
    ) -> None: ...

    @abstractmethod
    async def append_event(self, event: AgentEvent) -> None: ...


class PaperSearchGateway(ABC):
    @abstractmethod
    async def search(self, search_input: ArxivSearchInput) -> Sequence[Paper]: ...


class EventPublisher(ABC):
    @abstractmethod
    async def publish(
        self,
        event_type: str,
        payload: dict[str, JsonValue],
    ) -> None: ...
