import asyncio
from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import UUID, uuid4

import pytest

from app.application.run_registry import RunRegistry
from app.domain.enums import EventType
from app.domain.ports import ConversationStore
from app.domain.types import JsonValue
from app.infrastructure.agent.events import RunEventPublisher


async def _publisher(
    *,
    run_id: UUID | None = None,
    registry: RunRegistry | None = None,
) -> tuple[RunEventPublisher, RunRegistry, UUID, AsyncMock]:
    """Build a publisher for an already registered run, as ChatService does."""

    store = cast(AsyncMock, create_autospec(ConversationStore, instance=True))
    active_registry = registry or RunRegistry()
    active_run_id = run_id or uuid4()

    async def _noop() -> None:
        return None

    if active_registry.get(active_run_id) is None:
        active_registry.register(
            run_id=active_run_id,
            conversation_id=uuid4(),
            task=asyncio.create_task(_noop()),
        )
    publisher = RunEventPublisher(
        run_id=active_run_id,
        conversation_id=uuid4(),
        store=store,
        registry=active_registry,
    )
    return publisher, active_registry, active_run_id, store


@pytest.mark.asyncio
async def test_token_event_is_streamed_without_being_persisted() -> None:
    publisher, registry, run_id, store = await _publisher()

    await publisher.publish(EventType.TOKEN.value, {"text": "hello"})

    store.append_event.assert_not_awaited()
    active = registry.get(run_id)
    assert active is not None
    events, gap = active.snapshot(0)
    assert gap is False
    assert [event.type for event in events] == [EventType.TOKEN]
    assert events[0].payload == {"text": "hello"}


@pytest.mark.asyncio
async def test_persisted_tool_event_uses_compact_payload() -> None:
    publisher, registry, run_id, store = await _publisher()
    payload: dict[str, JsonValue] = {
        "tool_call_id": "call-1",
        "tool_name": "search_academic_papers",
        "result_count": 1,
        "duration_ms": 50,
        "papers": [{"title": "Large payload"}],
    }

    await publisher.publish(EventType.TOOL_COMPLETED.value, payload)

    assert store.append_event.await_args is not None
    persisted_event = store.append_event.await_args.args[0]
    assert "papers" not in persisted_event.payload
    assert persisted_event.payload["result_count"] == 1
    active = registry.get(run_id)
    assert active is not None
    events, _ = active.snapshot(0)
    assert "papers" in events[0].payload


@pytest.mark.asyncio
async def test_streamed_events_are_buffered_for_replay() -> None:
    publisher, registry, run_id, _ = await _publisher()

    await publisher.publish(EventType.RUN_STARTED.value, {"summary": "started"})
    await publisher.publish(EventType.TOKEN.value, {"text": "a"})
    await publisher.publish(EventType.TOKEN.value, {"text": "b"})

    active = registry.get(run_id)
    assert active is not None
    events, gap = active.snapshot(1)

    assert gap is False
    assert [event.payload for event in events] == [{"text": "a"}, {"text": "b"}]
    assert active.terminal_event is None
    assert active.is_finished is False


@pytest.mark.asyncio
async def test_finished_run_buffer_is_released_after_the_retention_window() -> None:
    """A finished run must not keep its token buffer for the process lifetime."""

    registry = RunRegistry(buffer_size=100, retention_seconds=0.01)
    publisher, _, run_id, _ = await _publisher(registry=registry)

    await publisher.publish(EventType.TOKEN.value, {"text": "a"})
    await publisher.publish(EventType.RUN_COMPLETED.value, {"run_id": str(run_id)})

    assert registry.get(run_id) is not None
    await asyncio.sleep(0.05)

    assert registry.get(run_id) is None
