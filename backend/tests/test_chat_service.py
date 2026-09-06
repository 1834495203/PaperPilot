import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from app.application.agent import AgentRunContext, AgentRunner
from app.application.chat_service import ChatService
from app.domain.entities import Message, RunMetrics
from app.domain.enums import EventType, RunStatus
from app.infrastructure.db.store import SqlAlchemyConversationStore


class _CompletedAgent(AgentRunner):
    async def run(
        self,
        *,
        history: Sequence[Message],
        context: AgentRunContext,
    ) -> RunMetrics:
        assert history[-1].metadata["run_id"] == str(context.run_id)
        return RunMetrics(
            input_tokens=12,
            output_tokens=4,
            total_tokens=16,
            llm_calls=1,
        )


class _BlockingAgent(AgentRunner):
    async def run(
        self,
        *,
        history: Sequence[Message],
        context: AgentRunContext,
    ) -> RunMetrics:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _FlakyAgent(AgentRunner):
    def __init__(self) -> None:
        self.calls = 0

    async def run(
        self,
        *,
        history: Sequence[Message],
        context: AgentRunContext,
    ) -> RunMetrics:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return RunMetrics(
            input_tokens=3,
            output_tokens=1,
            total_tokens=4,
            llm_calls=1,
        )


@pytest.mark.asyncio
async def test_each_user_request_is_linked_to_its_agent_run(tmp_path: Path) -> None:
    database_path = (tmp_path / "paperpilot-turn-test.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()
    service = ChatService(store, _CompletedAgent())
    conversation = await service.create_conversation("Turn")

    events = [
        event
        async for event in service.stream_message(
            conversation.id,
            "Who wrote this paper?",
            ["paper-1"],
        )
    ]
    messages = await service.get_messages(conversation.id)
    runs = await service.get_runs(conversation.id)

    assert len(runs) == 1
    assert messages[0].metadata == {
        "run_id": str(runs[0].id),
        "paper_ids": ["paper-1"],
    }
    assert {event.run_id for event in events} == {runs[0].id}
    assert runs[0].metrics.total_tokens == 16
    await store.close()


@pytest.mark.asyncio
async def test_active_run_can_be_cancelled(tmp_path: Path) -> None:
    database_path = (tmp_path / "paperpilot-cancel-test.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()
    service = ChatService(store, _BlockingAgent())
    conversation = await service.create_conversation("Cancel")
    stream = service.stream_message(conversation.id, "Stop this run")

    started = await anext(stream)
    assert started.type is EventType.RUN_STARTED
    assert await service.cancel_run(conversation.id, started.run_id) is True
    remaining = [event async for event in stream]
    runs = await service.get_runs(conversation.id)

    assert remaining[-1].type is EventType.RUN_CANCELLED
    assert runs[0].status is RunStatus.CANCELLED
    assert await service.cancel_run(conversation.id, started.run_id) is False
    await store.close()


@pytest.mark.asyncio
async def test_regenerate_replaces_target_turn(tmp_path: Path) -> None:
    database_path = (tmp_path / "paperpilot-regen-test.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()
    service = ChatService(store, _CompletedAgent())
    conversation = await service.create_conversation("Regen")

    _ = [
        event
        async for event in service.stream_message(
            conversation.id,
            "Question one",
            ["paper-1"],
        )
    ]
    messages = await service.get_messages(conversation.id)
    target = messages[0]
    old_run_id = target.metadata["run_id"]

    _ = [
        event
        async for event in service.regenerate_message(conversation.id, target.id)
    ]

    messages = await service.get_messages(conversation.id)
    runs = await service.get_runs(conversation.id)
    assert len(messages) == 1
    assert messages[0].content == "Question one"
    assert messages[0].metadata["paper_ids"] == ["paper-1"]
    assert len(runs) == 1
    assert messages[0].metadata["run_id"] == str(runs[0].id)
    assert str(runs[0].id) != old_run_id
    await store.close()


@pytest.mark.asyncio
async def test_regenerate_after_failed_run(tmp_path: Path) -> None:
    database_path = (tmp_path / "paperpilot-regen-fail-test.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()
    agent = _FlakyAgent()
    service = ChatService(store, agent)
    conversation = await service.create_conversation("Fail then retry")

    first_events = [
        event async for event in service.stream_message(conversation.id, "Question")
    ]
    assert first_events[-1].type is EventType.RUN_FAILED
    target = (await service.get_messages(conversation.id))[0]

    second_events = [
        event async for event in service.regenerate_message(conversation.id, target.id)
    ]

    assert second_events[-1].type is EventType.RUN_COMPLETED
    messages = await service.get_messages(conversation.id)
    runs = await service.get_runs(conversation.id)
    assert len(messages) == 1
    assert messages[0].content == "Question"
    assert len(runs) == 1
    assert runs[0].status is RunStatus.COMPLETED
    await store.close()
