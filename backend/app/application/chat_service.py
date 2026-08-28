import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from time import perf_counter
from uuid import UUID

from app.domain.entities import AgentEvent, Conversation, Message, RunMetrics
from app.domain.enums import EventType, MessageRole, RunStatus
from app.domain.ports import ConversationStore
from app.infrastructure.agent.events import PersistentQueueEventPublisher
from app.infrastructure.agent.graph import GraphRunContext, PaperAgentGraph


class ConversationNotFoundError(LookupError):
    pass


class ChatService:
    def __init__(self, store: ConversationStore, agent: PaperAgentGraph) -> None:
        self._store = store
        self._agent = agent

    async def create_conversation(self, title: str) -> Conversation:
        return await self._store.create_conversation(title)

    async def list_conversations(self) -> Sequence[Conversation]:
        return await self._store.list_conversations()

    async def get_messages(self, conversation_id: UUID) -> Sequence[Message]:
        await self._require_conversation(conversation_id)
        return await self._store.list_messages(conversation_id)

    async def stream_message(
        self,
        conversation_id: UUID,
        content: str,
    ) -> AsyncIterator[AgentEvent]:
        await self._require_conversation(conversation_id)
        await self._store.append_message(conversation_id, MessageRole.USER, content)
        run = await self._store.create_run(conversation_id)
        queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        publisher = PersistentQueueEventPublisher(
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
        publisher: PersistentQueueEventPublisher,
    ) -> None:
        started = perf_counter()
        await publisher.publish(
            EventType.RUN_STARTED.value,
            {"run_id": str(run_id), "summary": "开始执行单 Agent 文献研究任务"},
        )
        try:
            history = await self._store.list_messages(conversation_id)
            metrics = await self._agent.run(
                history=history,
                context=GraphRunContext(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    publisher=publisher,
                ),
            )
            metrics = replace(metrics, duration_ms=int((perf_counter() - started) * 1000))
            await self._store.finish_run(run_id, RunStatus.COMPLETED, metrics)
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
                },
            )
        except Exception as error:
            metrics = RunMetrics(duration_ms=int((perf_counter() - started) * 1000))
            await self._store.finish_run(run_id, RunStatus.FAILED, metrics, str(error))
            await publisher.publish(
                EventType.RUN_FAILED.value,
                {
                    "run_id": str(run_id),
                    "duration_ms": metrics.duration_ms,
                    "error": str(error),
                },
            )

    async def _require_conversation(self, conversation_id: UUID) -> Conversation:
        conversation = await self._store.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(str(conversation_id))
        return conversation
