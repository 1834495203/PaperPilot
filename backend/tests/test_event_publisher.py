import asyncio
from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

import pytest

from app.domain.entities import AgentEvent
from app.domain.enums import EventType
from app.domain.ports import ConversationStore
from app.domain.types import JsonValue
from app.infrastructure.agent.events import RunEventPublisher


@pytest.mark.asyncio
async def test_token_event_is_streamed_without_being_persisted() -> None:
    store = cast(ConversationStore, create_autospec(ConversationStore, instance=True))
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    publisher = RunEventPublisher(
        run_id=uuid4(),
        conversation_id=uuid4(),
        store=store,
        queue=queue,
    )

    await publisher.publish(EventType.TOKEN.value, {"text": "hello"})

    cast(AsyncMock, store.append_event).assert_not_awaited()
    queued_event = await queue.get()
    assert queued_event.type is EventType.TOKEN
    assert queued_event.payload == {"text": "hello"}


@pytest.mark.asyncio
async def test_persisted_tool_event_uses_compact_payload() -> None:
    store = cast(ConversationStore, create_autospec(ConversationStore, instance=True))
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    publisher = RunEventPublisher(
        run_id=uuid4(),
        conversation_id=uuid4(),
        store=store,
        queue=queue,
    )
    payload: dict[str, JsonValue] = {
        "tool_call_id": "call-1",
        "tool_name": "search_academic_papers",
        "result_count": 1,
        "duration_ms": 50,
        "papers": [{"title": "Large payload"}],
    }

    await publisher.publish(EventType.TOOL_COMPLETED.value, payload)

    append_event = cast(AsyncMock, store.append_event)
    assert append_event.await_args is not None
    persisted_event = append_event.await_args.args[0]
    assert "papers" not in persisted_event.payload
    assert persisted_event.payload["result_count"] == 1
    queued_event = await queue.get()
    assert "papers" in queued_event.payload
