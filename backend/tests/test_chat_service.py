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
