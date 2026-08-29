from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from app.domain.entities import AgentEvent, RunMetrics
from app.domain.enums import EventType, MessageRole, RunStatus
from app.infrastructure.db.store import SqlAlchemyConversationStore


@pytest.mark.asyncio
async def test_messages_are_persisted_in_sequence(tmp_path: Path) -> None:
    database_path = (tmp_path / "paperpilot-test.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()

    conversation = await store.create_conversation("Test")
    first = await store.append_message(conversation.id, MessageRole.USER, "Find RAG papers")
    second = await store.append_message(conversation.id, MessageRole.ASSISTANT, "Searching")
    messages = await store.list_messages(conversation.id)

    assert first.sequence == 1
    assert second.sequence == 2
    assert [message.content for message in messages] == ["Find RAG papers", "Searching"]
    await store.close()


@pytest.mark.asyncio
async def test_conversation_metrics_refresh_from_run_history(tmp_path: Path) -> None:
    database_path = (tmp_path / "paperpilot-metrics-test.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()
    conversation = await store.create_conversation("Metrics")

    first_run = await store.create_run(conversation.id)
    await store.finish_run(
        first_run.id,
        RunStatus.COMPLETED,
        RunMetrics(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            llm_calls=2,
            tool_calls=1,
            duration_ms=800,
        ),
    )
    first_snapshot = await store.refresh_conversation_metrics(conversation.id)

    second_run = await store.create_run(conversation.id)
    await store.finish_run(
        second_run.id,
        RunStatus.COMPLETED,
        RunMetrics(
            input_tokens=60,
            output_tokens=20,
            total_tokens=80,
            llm_calls=1,
            tool_calls=0,
            duration_ms=300,
        ),
    )
    accumulated = await store.refresh_conversation_metrics(conversation.id)

    assert first_snapshot.total_tokens == 140
    assert accumulated.total_tokens == 220
    assert accumulated.total_duration_ms == 1100
    assert accumulated.llm_calls == 3
    assert accumulated.tool_calls == 1
    assert accumulated.run_count == 2
    await store.close()


@pytest.mark.asyncio
async def test_list_events_excludes_legacy_token_events(tmp_path: Path) -> None:
    database_path = (tmp_path / "paperpilot-events-test.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()
    conversation = await store.create_conversation("Events")
    run = await store.create_run(conversation.id)

    await store.append_event(
        AgentEvent.create(
            run_id=run.id,
            conversation_id=conversation.id,
            sequence=1,
            event_type=EventType.TOKEN,
            payload={"text": "legacy token"},
        )
    )
    persisted_event = AgentEvent(
        id=uuid4(),
        run_id=run.id,
        conversation_id=conversation.id,
        sequence=2,
        type=EventType.STAGE_STARTED,
        timestamp=run.started_at,
        payload={"summary": "Analyzing"},
    )
    await store.append_event(persisted_event)

    events = await store.list_events(conversation.id, limit=100)

    assert [event.type for event in events] == [EventType.STAGE_STARTED]
    await store.close()
