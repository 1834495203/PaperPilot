import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from time import perf_counter
from uuid import UUID

from app.application.agent import AgentRunContext, AgentRunner
from app.application.run_registry import ActiveRun, RunRegistry, resumed_event_payload
from app.domain.entities import (
    AgentEvent,
    AgentRun,
    Conversation,
    ConversationMetrics,
    Message,
    RunMetrics,
)
from app.domain.enums import TERMINAL_EVENT_TYPES, EventType, MessageRole, RunStatus
from app.domain.ports import ConversationStore
from app.infrastructure.agent.events import RunEventPublisher


class ConversationNotFoundError(LookupError):
    pass


class MessageNotFoundError(LookupError):
    pass


class ConversationBusyError(RuntimeError):
    """Another run is already executing in this conversation."""

    def __init__(self, conversation_id: UUID, run_id: UUID) -> None:
        super().__init__(
            f"Conversation {conversation_id} already has an active run {run_id}; "
            "stop or finish it before starting another one"
        )
        self.conversation_id = conversation_id
        self.run_id = run_id


class ChatService:
    """Owns conversation state and the lifecycle of agent runs.

    A run executes as an independent task inside a :class:`RunRegistry`, so losing
    the streaming connection only detaches a reader. Reconnecting replays the
    buffered events after the last sequence the client saw.
    """

    def __init__(
        self,
        store: ConversationStore,
        agent: AgentRunner,
        *,
        registry: RunRegistry | None = None,
    ) -> None:
        self._store = store
        self._agent = agent
        self._registry = registry or RunRegistry()

    @property
    def registry(self) -> RunRegistry:
        return self._registry

    async def create_conversation(self, title: str) -> Conversation:
        return await self._store.create_conversation(title)

    async def list_conversations(self) -> Sequence[Conversation]:
        return await self._store.list_conversations()

    async def delete_conversation(self, conversation_id: UUID) -> None:
        active_run_id = self._registry.active_run_id(conversation_id)
        if active_run_id is not None:
            raise ConversationBusyError(conversation_id, active_run_id)
        deleted = await self._store.delete_conversation(conversation_id)
        if not deleted:
            raise ConversationNotFoundError(str(conversation_id))

    async def get_messages(
        self,
        conversation_id: UUID,
        *,
        include_superseded: bool = False,
    ) -> Sequence[Message]:
        await self._require_conversation(conversation_id)
        return await self._store.list_messages(
            conversation_id,
            include_superseded=include_superseded,
        )

    async def get_conversation_metrics(
        self,
        conversation_id: UUID,
    ) -> ConversationMetrics:
        await self._require_conversation(conversation_id)
        return await self._store.get_conversation_metrics(conversation_id)

    async def get_runs(self, conversation_id: UUID) -> Sequence[AgentRun]:
        await self._require_conversation(conversation_id)
        return await self._store.list_runs(conversation_id)

    async def get_events(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> Sequence[AgentEvent]:
        await self._require_conversation(conversation_id)
        return await self._store.list_events(conversation_id, limit)

    async def active_run_id(self, conversation_id: UUID) -> UUID | None:
        return self._registry.active_run_id(conversation_id)

    async def cancel_run(self, conversation_id: UUID, run_id: UUID) -> bool:
        await self._require_conversation(conversation_id)
        active = self._registry.get(run_id)
        if active is None or active.conversation_id != conversation_id:
            return False
        if active.task.done():
            return False
        return await self._registry.cancel(run_id)

    async def stream_message(
        self,
        conversation_id: UUID,
        content: str,
        paper_ids: Sequence[str] = (),
        *,
        local_corpus_available: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        await self._require_conversation(conversation_id)
        self._require_idle_conversation(conversation_id)
        selected_paper_ids = tuple(dict.fromkeys(paper_ids))
        run = await self._store.create_run(conversation_id)
        await self._store.append_message(
            conversation_id,
            MessageRole.USER,
            content,
            metadata={
                "run_id": str(run.id),
                "paper_ids": list(selected_paper_ids),
            },
        )
        active = self._start_run(
            conversation_id=conversation_id,
            run_id=run.id,
            paper_ids=selected_paper_ids,
            local_corpus_available=local_corpus_available,
        )
        async for event in self._subscribe(active, after_sequence=0):
            yield event

    async def regenerate_message(
        self,
        conversation_id: UUID,
        message_id: UUID,
        *,
        local_corpus_available: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Answer a turn again without destroying the answer being replaced.

        The replaced turn is superseded only after the new answer succeeds, so a
        failed or cancelled retry leaves the conversation exactly as it was, and the
        previous version stays readable through ``include_superseded``.
        """

        await self._require_conversation(conversation_id)
        self._require_idle_conversation(conversation_id)
        target = await self._store.get_message(conversation_id, message_id)
        if target is None:
            raise MessageNotFoundError(str(message_id))
        question, previous = await self._turn_being_regenerated(conversation_id, target)
        raw_paper_ids = question.metadata.get("paper_ids")
        selected_paper_ids = (
            tuple(dict.fromkeys(str(item) for item in raw_paper_ids))
            if isinstance(raw_paper_ids, list)
            else ()
        )
        replaced_run_id = _run_id_of(previous) or _run_id_of(question)
        run = await self._store.create_run(conversation_id)
        attempt = await self._store.append_message(
            conversation_id,
            MessageRole.USER,
            question.content,
            metadata={
                "run_id": str(run.id),
                "paper_ids": list(selected_paper_ids),
                "regenerates_message_id": str(target.id),
            },
        )
        active = self._start_run(
            conversation_id=conversation_id,
            run_id=run.id,
            paper_ids=selected_paper_ids,
            local_corpus_available=local_corpus_available,
        )
        try:
            async for event in self._subscribe(active, after_sequence=0):
                yield event
        finally:
            await self._settle_regeneration(
                conversation_id=conversation_id,
                run_id=run.id,
                question=question,
                attempt_sequence=attempt.sequence,
                replaced_run_id=(
                    replaced_run_id if replaced_run_id != str(run.id) else None
                ),
            )

    async def resume_run(
        self,
        conversation_id: UUID,
        run_id: UUID,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[AgentEvent]:
        """Re-attach to a run: buffered replay when live, persisted replay when not."""

        await self._require_conversation(conversation_id)
        active = self._registry.get(run_id)
        if active is not None and active.conversation_id == conversation_id:
            async for event in self._subscribe(active, after_sequence=after_sequence):
                yield event
            return
        run = await self._store.get_run(conversation_id, run_id)
        if run is None:
            raise MessageNotFoundError(str(run_id))
        events = await self._store.list_run_events(run_id, after_sequence=after_sequence)
        yield AgentEvent.create(
            run_id=run_id,
            conversation_id=conversation_id,
            sequence=after_sequence,
            event_type=EventType.RUN_RESUMED,
            payload=resumed_event_payload(
                run_id=run_id,
                replayed=len(events),
                gap=True,
                active=run.status is RunStatus.RUNNING,
                latest_sequence=events[-1].sequence if events else after_sequence,
            ),
        )
        for event in events:
            yield event

    def _start_run(
        self,
        *,
        conversation_id: UUID,
        run_id: UUID,
        paper_ids: tuple[str, ...],
        local_corpus_available: bool,
    ) -> ActiveRun:
        publisher = RunEventPublisher(
            run_id=run_id,
            conversation_id=conversation_id,
            store=self._store,
            registry=self._registry,
        )
        task = asyncio.create_task(
            self._execute_run(
                conversation_id=conversation_id,
                run_id=run_id,
                publisher=publisher,
                paper_ids=paper_ids,
                local_corpus_available=local_corpus_available,
            ),
            name=f"paperpilot-run-{run_id}",
        )
        return self._registry.register(
            run_id=run_id,
            conversation_id=conversation_id,
            task=task,
        )

    async def _subscribe(
        self,
        active: ActiveRun,
        *,
        after_sequence: int,
    ) -> AsyncIterator[AgentEvent]:
        """Stream a run from a sequence, then follow it live.

        Closing this generator only detaches the reader: the run keeps executing and
        its events stay buffered for the next subscriber.
        """

        replayed, gap = active.snapshot(after_sequence)
        if after_sequence > 0 or gap:
            yield AgentEvent.create(
                run_id=active.run_id,
                conversation_id=active.conversation_id,
                sequence=after_sequence,
                event_type=EventType.RUN_RESUMED,
                payload=resumed_event_payload(
                    run_id=active.run_id,
                    replayed=len(replayed),
                    gap=gap,
                    active=not active.is_finished,
                    latest_sequence=(replayed[-1].sequence if replayed else after_sequence),
                ),
            )
        for event in replayed:
            yield event
        if active.is_finished:
            return
        subscription = active.attach()
        try:
            while True:
                live_event = await subscription.queue.get()
                if live_event is None:
                    return
                yield live_event
                if live_event.type in TERMINAL_EVENT_TYPES:
                    return
        finally:
            active.detach(subscription)

    async def _settle_regeneration(
        self,
        *,
        conversation_id: UUID,
        run_id: UUID,
        question: Message,
        attempt_sequence: int,
        replaced_run_id: str | None,
    ) -> None:
        """Decide which version of a regenerated turn stays live.

        The replaced turn keeps its rows and is marked superseded, so the previous
        answer stays readable. Only a superseded turn's run record - not its
        messages - is removed, which keeps the trace list free of duplicate turns.
        """

        active = self._registry.get(run_id)
        succeeded = (
            active is not None
            and active.terminal_event is not None
            and active.terminal_event.type is EventType.RUN_COMPLETED
        )
        if not succeeded:
            # The attempt lost, so the turn it replaced stays live and the question
            # this attempt appended is hidden rather than deleted.
            await self._store.supersede_run_messages(
                conversation_id,
                run_id,
                superseded_by_run=run_id,
            )
            return
        await self._store.supersede_messages_from(
            conversation_id,
            question.sequence,
            superseded_by_run=run_id,
            before_sequence=attempt_sequence,
        )
        if replaced_run_id is not None:
            try:
                await self._store.delete_run(UUID(replaced_run_id))
            except ValueError:
                return

    async def _turn_being_regenerated(
        self,
        conversation_id: UUID,
        target: Message,
    ) -> tuple[Message, Message | None]:
        """Resolve a regenerate target to its question and the answer it replaces."""

        messages = await self._store.list_messages(
            conversation_id,
            include_superseded=True,
        )
        if target.role is MessageRole.USER:
            previous = next(
                (
                    message
                    for message in messages
                    if message.sequence > target.sequence
                    and message.role is MessageRole.ASSISTANT
                    and message.is_active
                ),
                None,
            )
            return target, previous
        question = next(
            (
                message
                for message in reversed(messages)
                if message.sequence < target.sequence
                and message.role is MessageRole.USER
            ),
            None,
        )
        if question is None:
            raise MessageNotFoundError(str(target.id))
        return question, target

    def _require_idle_conversation(self, conversation_id: UUID) -> None:
        active_run_id = self._registry.active_run_id(conversation_id)
        if active_run_id is not None:
            raise ConversationBusyError(conversation_id, active_run_id)

    async def _execute_run(
        self,
        *,
        conversation_id: UUID,
        run_id: UUID,
        publisher: RunEventPublisher,
        paper_ids: tuple[str, ...],
        local_corpus_available: bool,
    ) -> None:
        started = perf_counter()
        metrics_accumulated = False
        await publisher.publish(
            EventType.RUN_STARTED.value,
            {"run_id": str(run_id), "summary": "开始执行 Supervisor 多 Agent 研究任务"},
        )
        try:
            history = await self._store.list_messages(conversation_id)
            metrics = await self._agent.run(
                history=history,
                context=AgentRunContext(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    publisher=publisher,
                    paper_ids=paper_ids,
                    local_corpus_available=local_corpus_available,
                ),
            )
            metrics = replace(metrics, duration_ms=int((perf_counter() - started) * 1000))
            await self._store.finish_run(run_id, RunStatus.COMPLETED, metrics)
            conversation_metrics = await self._store.refresh_conversation_metrics(
                conversation_id
            )
            metrics_accumulated = True
            await publisher.publish(
                EventType.RUN_COMPLETED.value,
                {
                    "run_id": str(run_id),
                    "duration_ms": metrics.duration_ms,
                    "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "total_tokens": metrics.total_tokens,
                    "llm_calls": metrics.llm_calls,
                    "tool_calls": metrics.tool_calls,
                    "conversation_input_tokens": conversation_metrics.input_tokens,
                    "conversation_output_tokens": conversation_metrics.output_tokens,
                    "conversation_total_tokens": conversation_metrics.total_tokens,
                    "conversation_llm_calls": conversation_metrics.llm_calls,
                    "conversation_tool_calls": conversation_metrics.tool_calls,
                    "conversation_total_duration_ms": conversation_metrics.total_duration_ms,
                    "conversation_run_count": conversation_metrics.run_count,
                },
            )
        except asyncio.CancelledError:
            metrics = RunMetrics(duration_ms=int((perf_counter() - started) * 1000))
            await self._store.finish_run(
                run_id,
                RunStatus.CANCELLED,
                metrics,
                "Run cancelled by the user",
            )
            conversation_metrics = await self._store.refresh_conversation_metrics(
                conversation_id
            )
            await publisher.publish(
                EventType.RUN_CANCELLED.value,
                {
                    "run_id": str(run_id),
                    "duration_ms": metrics.duration_ms,
                    "summary": "任务已停止",
                    "conversation_input_tokens": conversation_metrics.input_tokens,
                    "conversation_output_tokens": conversation_metrics.output_tokens,
                    "conversation_total_tokens": conversation_metrics.total_tokens,
                    "conversation_llm_calls": conversation_metrics.llm_calls,
                    "conversation_tool_calls": conversation_metrics.tool_calls,
                    "conversation_total_duration_ms": conversation_metrics.total_duration_ms,
                    "conversation_run_count": conversation_metrics.run_count,
                },
            )
        except Exception as error:
            metrics = RunMetrics(duration_ms=int((perf_counter() - started) * 1000))
            await self._store.finish_run(run_id, RunStatus.FAILED, metrics, str(error))
            conversation_metrics = await self._store.get_conversation_metrics(conversation_id)
            if not metrics_accumulated:
                conversation_metrics = await self._store.refresh_conversation_metrics(
                    conversation_id
                )
            await publisher.publish(
                EventType.RUN_FAILED.value,
                {
                    "run_id": str(run_id),
                    "duration_ms": metrics.duration_ms,
                    "error": str(error),
                    "conversation_input_tokens": conversation_metrics.input_tokens,
                    "conversation_output_tokens": conversation_metrics.output_tokens,
                    "conversation_total_tokens": conversation_metrics.total_tokens,
                    "conversation_llm_calls": conversation_metrics.llm_calls,
                    "conversation_tool_calls": conversation_metrics.tool_calls,
                    "conversation_total_duration_ms": conversation_metrics.total_duration_ms,
                    "conversation_run_count": conversation_metrics.run_count,
                },
            )

    async def _require_conversation(self, conversation_id: UUID) -> Conversation:
        conversation = await self._store.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(str(conversation_id))
        return conversation


def _run_id_of(message: Message | None) -> str | None:
    if message is None:
        return None
    value = message.metadata.get("run_id")
    return value if isinstance(value, str) else None
