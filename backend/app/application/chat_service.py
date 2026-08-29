import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from time import perf_counter
from uuid import UUID

from app.application.agent import AgentRunContext, AgentRunner
from app.domain.entities import (
    AgentEvent,
    Conversation,
    ConversationMetrics,
    Message,
    RunMetrics,
)
from app.domain.enums import EventType, MessageRole, RunStatus
from app.domain.ports import ConversationStore
from app.infrastructure.agent.events import RunEventPublisher


class ConversationNotFoundError(LookupError):
    pass


class ChatService:
    def __init__(self, store: ConversationStore, agent: AgentRunner) -> None:
        self._store = store
        self._agent = agent

    async def create_conversation(self, title: str) -> Conversation:
        return await self._store.create_conversation(title)

    async def list_conversations(self) -> Sequence[Conversation]:
        return await self._store.list_conversations()

    async def get_messages(self, conversation_id: UUID) -> Sequence[Message]:
        await self._require_conversation(conversation_id)
        return await self._store.list_messages(conversation_id)

    async def get_conversation_metrics(
        self,
        conversation_id: UUID,
    ) -> ConversationMetrics:
        await self._require_conversation(conversation_id)
        return await self._store.get_conversation_metrics(conversation_id)

    async def get_events(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> Sequence[AgentEvent]:
        await self._require_conversation(conversation_id)
        return await self._store.list_events(conversation_id, limit)

    async def stream_message(
        self,
        conversation_id: UUID,
        content: str,
    ) -> AsyncIterator[AgentEvent]:
        await self._require_conversation(conversation_id)
        await self._store.append_message(conversation_id, MessageRole.USER, content)
        run = await self._store.create_run(conversation_id)
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        publisher = RunEventPublisher(
            run_id=run.id,
            conversation_id=conversation_id,
            store=self._store,
            queue=queue,
        )
        task = asyncio.create_task(
            self._execute_run(
                conversation_id=conversation_id,
                run_id=run.id,
                publisher=publisher,
            ),
            name=f"paperpilot-run-{run.id}",
        )

        try:
            while True:
                event = await queue.get()
                yield event
                if event.type in {EventType.RUN_COMPLETED, EventType.RUN_FAILED}:
                    break
            await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    metrics = RunMetrics()
                    await self._store.finish_run(
                        run.id,
                        RunStatus.FAILED,
                        metrics,
                        "Client disconnected before the run completed",
                    )

    async def _execute_run(
        self,
        *,
        conversation_id: UUID,
        run_id: UUID,
        publisher: RunEventPublisher,
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
