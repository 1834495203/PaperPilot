"""Run-level reliability: resumable runs and non-destructive regeneration.

Two acceptance properties are covered here:

- losing the streaming connection must not cancel the run, and the client must be
  able to resume from the last sequence it saw;
- regenerating an answer must not destroy the answer it replaces, and a failed
  retry must leave the previous turn exactly as it was.
"""

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from app.application.agent import AgentRunContext, AgentRunner
from app.application.chat_service import ChatService, ConversationBusyError
from app.domain.entities import AgentEvent, AgentRun, Conversation, Message, RunMetrics
from app.domain.enums import EventType, MessageRole
from app.domain.ports import ConversationStore
from app.infrastructure.db.store import SqlAlchemyConversationStore


class _AnsweringAgent(AgentRunner):
    """Records an assistant message the way the real Writer does."""

    def __init__(self, store: ConversationStore, answer: str = "First answer") -> None:
        self._store = store
        self.answer = answer
        self.calls = 0

    async def run(
        self,
        *,
        history: Sequence[Message],
        context: AgentRunContext,
    ) -> RunMetrics:
        del history
        self.calls += 1
        await context.publisher.publish(
            EventType.TOKEN.value,
            {"text": self.answer, "source": "model", "actor": "writer"},
        )
        await self._store.append_message(
            context.conversation_id,
            MessageRole.ASSISTANT,
            self.answer,
            {"run_id": str(context.run_id), "citations": []},
        )
        return RunMetrics(total_tokens=1, llm_calls=1)


class _SlowAgent(AgentRunner):
    """Signals when it started and waits, so a test can drop the stream."""

    def __init__(self, store: ConversationStore) -> None:
        self._store = store
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = False

    async def run(
        self,
        *,
        history: Sequence[Message],
        context: AgentRunContext,
    ) -> RunMetrics:
        del history
        await context.publisher.publish(EventType.TOKEN.value, {"text": "partial"})
        self.started.set()
        await self.release.wait()
        await context.publisher.publish(EventType.TOKEN.value, {"text": "done"})
        await self._store.append_message(
            context.conversation_id,
            MessageRole.ASSISTANT,
            "partial done",
            {"run_id": str(context.run_id)},
        )
        self.finished = True
        return RunMetrics(total_tokens=2, llm_calls=1)


class _FailingAgent(AgentRunner):
    async def run(
        self,
        *,
        history: Sequence[Message],
        context: AgentRunContext,
    ) -> RunMetrics:
        del history, context
        raise RuntimeError("model unavailable")


def _tracking(
    created: list[AgentRunner],
    builder: Callable[[ConversationStore], AgentRunner],
) -> Callable[[ConversationStore], AgentRunner]:
    """Wrap a builder so the test keeps a handle on the agent it created."""

    def build(store: ConversationStore) -> AgentRunner:
        agent = builder(store)
        created.append(agent)
        return agent

    return build


async def _setup(
    tmp_path: Path,
    name: str,
    agent_factory: Callable[[ConversationStore], AgentRunner],
) -> tuple[ChatService, SqlAlchemyConversationStore, Conversation]:
    database_path = (tmp_path / f"{name}.db").as_posix()
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    store = SqlAlchemyConversationStore(engine)
    await store.initialize()
    service = ChatService(store, agent_factory(store))
    conversation = await service.create_conversation(name)
    return service, store, conversation


@pytest.mark.asyncio
async def test_regenerate_keeps_the_previous_answer_as_a_version(tmp_path: Path) -> None:
    created: list[AgentRunner] = []
    service, store, conversation = await _setup(
        tmp_path,
        "versions",
        _tracking(created, lambda conversation_store: _AnsweringAgent(conversation_store)),
    )
    agent = cast(_AnsweringAgent, created[0])

    _ = [event async for event in service.stream_message(conversation.id, "Question")]
    first_turn = await service.get_messages(conversation.id)
    first_answer = next(
        message for message in first_turn if message.role is MessageRole.ASSISTANT
    )

    agent.answer = "Second answer"
    _ = [
        event
        async for event in service.regenerate_message(conversation.id, first_turn[0].id)
    ]

    active = await service.get_messages(conversation.id)
    everything = await service.get_messages(conversation.id, include_superseded=True)

    assert [message.role for message in active] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert active[1].content == "Second answer"
    assert active[1].is_active is True
    assert len(everything) == 4
    superseded = [message for message in everything if not message.is_active]
    assert {message.content for message in superseded} == {"Question", "First answer"}
    assert all(message.superseded_by_run is not None for message in superseded)
    assert first_answer.id in {message.id for message in superseded}
    runs = await service.get_runs(conversation.id)
    assert len(runs) == 1
    await store.close()


@pytest.mark.asyncio
async def test_failed_regenerate_keeps_the_previous_answer_live(tmp_path: Path) -> None:
    created: list[AgentRunner] = []
    service, store, conversation = await _setup(
        tmp_path,
        "keep-on-failure",
        _tracking(created, lambda conversation_store: _AnsweringAgent(conversation_store)),
    )

    _ = [event async for event in service.stream_message(conversation.id, "Question")]
    turn = await service.get_messages(conversation.id)

    service._agent = _FailingAgent()  # noqa: SLF001
    events = [
        event async for event in service.regenerate_message(conversation.id, turn[0].id)
    ]

    assert events[-1].type is EventType.RUN_FAILED
    active = await service.get_messages(conversation.id)
    everything = await service.get_messages(conversation.id, include_superseded=True)

    assert [message.content for message in active] == ["Question", "First answer"]
    assert all(message.is_active for message in active)
    hidden = [message for message in everything if not message.is_active]
    assert [message.content for message in hidden] == ["Question"]
    await store.close()


@pytest.mark.asyncio
async def test_run_survives_a_dropped_subscription_and_can_be_resumed(
    tmp_path: Path,
) -> None:
    created: list[AgentRunner] = []
    service, store, conversation = await _setup(
        tmp_path,
        "resume",
        _tracking(created, lambda conversation_store: _SlowAgent(conversation_store)),
    )
    agent = cast(_SlowAgent, created[0])

    stream = service.stream_message(conversation.id, "Question")
    seen: list[AgentEvent] = []
    async for event in stream:
        seen.append(event)
        if event.type is EventType.TOKEN:
            break
    # The reader is gone; the run must keep going.
    await stream.aclose()
    active_run_id = service.registry.active_run_id(conversation.id)
    assert active_run_id is not None

    agent.release.set()
    resumed = [
        event
        async for event in service.resume_run(
            conversation.id,
            active_run_id,
            after_sequence=seen[-1].sequence,
        )
    ]

    assert resumed[0].type is EventType.RUN_RESUMED
    assert resumed[0].payload["gap"] is False
    assert [event.payload["text"] for event in resumed if event.type is EventType.TOKEN] == [
        "done"
    ]
    assert resumed[-1].type is EventType.RUN_COMPLETED
    assert agent.finished is True
    messages = await service.get_messages(conversation.id)
    assert messages[-1].content == "partial done"
    await store.close()


@pytest.mark.asyncio
async def test_finished_run_can_still_be_replayed_after_a_restart(tmp_path: Path) -> None:
    """A new service instance has no buffer, so replay falls back to the store."""

    service, store, conversation = await _setup(
        tmp_path,
        "cold-resume",
        lambda conversation_store: _AnsweringAgent(conversation_store, "Answer"),
    )

    _ = [event async for event in service.stream_message(conversation.id, "Question")]
    runs: Sequence[AgentRun] = await service.get_runs(conversation.id)

    cold_service = ChatService(store, _AnsweringAgent(store, "Answer"))
    events = [
        event async for event in cold_service.resume_run(conversation.id, runs[0].id)
    ]

    assert events[0].type is EventType.RUN_RESUMED
    assert events[0].payload["gap"] is True
    assert events[0].payload["active"] is False
    assert events[-1].type is EventType.RUN_COMPLETED
    # Token events are streamed only, so a cold replay cannot contain them.
    assert all(event.type is not EventType.TOKEN for event in events)
    await store.close()


@pytest.mark.asyncio
async def test_second_run_in_one_conversation_is_refused(tmp_path: Path) -> None:
    created: list[AgentRunner] = []
    service, store, conversation = await _setup(
        tmp_path,
        "busy",
        _tracking(created, lambda conversation_store: _SlowAgent(conversation_store)),
    )
    agent = cast(_SlowAgent, created[0])

    stream = service.stream_message(conversation.id, "First question")
    async for event in stream:
        if event.type is EventType.TOKEN:
            break

    with pytest.raises(ConversationBusyError):
        _ = [event async for event in service.stream_message(conversation.id, "Second")]
    with pytest.raises(ConversationBusyError):
        await service.delete_conversation(conversation.id)

    await stream.aclose()
    agent.release.set()
    active_run_id = service.registry.active_run_id(conversation.id)
    assert active_run_id is not None
    async for _ in service.resume_run(conversation.id, active_run_id):
        pass

    assert service.registry.active_run_id(conversation.id) is None
    await store.close()
