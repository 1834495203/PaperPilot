import asyncio
from uuid import UUID

from app.domain.entities import AgentEvent
from app.domain.enums import EventType
from app.domain.ports import ConversationStore, EventPublisher
from app.domain.types import JsonValue


class PersistentQueueEventPublisher(EventPublisher):
    def __init__(
        self,
        *,
        run_id: UUID,
        conversation_id: UUID,
        store: ConversationStore,
        queue: asyncio.Queue[AgentEvent],
    ) -> None:
        self._run_id = run_id
        self._conversation_id = conversation_id
        self._store = store
        self._queue = queue
        self._sequence = 0

    async def publish(
        self,
        event_type: str,
        payload: dict[str, JsonValue],
    ) -> None:
        self._sequence += 1
        event = AgentEvent.create(
            run_id=self._run_id,
            conversation_id=self._conversation_id,
            sequence=self._sequence,
            event_type=EventType(event_type),
            payload=payload,
        )
        await self._store.append_event(event)
        await self._queue.put(event)
