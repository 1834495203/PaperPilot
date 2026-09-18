import logging
from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import UUID

from app.domain.entities import AgentEvent
from app.domain.enums import PERSISTED_AGENT_EVENT_TYPES, EventType
from app.domain.ports import ConversationStore, EventPublisher
from app.domain.types import JsonValue

if TYPE_CHECKING:
    from app.application.run_registry import RunRegistry


class EventPersistencePolicy:
    def prepare_for_storage(self, event: AgentEvent) -> AgentEvent | None:
        if event.type not in PERSISTED_AGENT_EVENT_TYPES:
            return None
        if event.type is not EventType.TOOL_COMPLETED:
            return event

        compact_payload = {
            key: value
            for key, value in event.payload.items()
            if key
            in {
                "tool_call_id",
                "tool_name",
                "result_count",
                "initial_hit_count",
                "expanded_candidate_count",
                "hit_count",
                "extracted_pages",
                "duration_ms",
            }
        }
        return replace(event, payload=compact_payload)


class RunEventPublisher(EventPublisher):
    def __init__(
        self,
        *,
        run_id: UUID,
        conversation_id: UUID,
        store: ConversationStore,
        registry: "RunRegistry",
        persistence_policy: EventPersistencePolicy | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._run_id = run_id
        self._conversation_id = conversation_id
        self._store = store
        self._registry = registry
        self._persistence_policy = persistence_policy or EventPersistencePolicy()
        self._logger = logger or logging.getLogger(__name__)
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
        stored_event = self._persistence_policy.prepare_for_storage(event)
        if stored_event is not None:
            await self._store.append_event(stored_event)
        self._write_log(event)
        # Fan out through the registry instead of one queue, so several readers can
        # follow the same run and a reconnect can replay what it missed.
        self._registry.publish(self._run_id, event)

    def _write_log(self, event: AgentEvent) -> None:
        if event.type is EventType.TOKEN:
            return
        log_level = logging.ERROR if event.type is EventType.RUN_FAILED else logging.INFO
        self._logger.log(
            log_level,
            "agent_event type=%s run_id=%s sequence=%d",
            event.type.value,
            event.run_id,
            event.sequence,
        )
