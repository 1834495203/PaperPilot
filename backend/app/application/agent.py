from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from app.domain.entities import Message, RunMetrics
from app.domain.ports import EventPublisher


@dataclass(frozen=True, slots=True)
class AgentRunContext:
    """Per-run services and identifiers that must not be stored in graph state."""

    conversation_id: UUID
    run_id: UUID
    publisher: EventPublisher
    paper_ids: tuple[str, ...] = ()
    local_corpus_available: bool = False


class AgentRunner(ABC):
    """Application-facing contract implemented by any conversational agent."""

    @abstractmethod
    async def run(
        self,
        *,
        history: Sequence[Message],
        context: AgentRunContext,
    ) -> RunMetrics: ...
