from collections.abc import Sequence
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.domain.entities import (
    AgentEvent,
    AgentRun,
    Conversation,
    ConversationMetrics,
    Message,
    RunMetrics,
    utc_now,
)
from app.domain.enums import PERSISTED_AGENT_EVENT_TYPES, EventType, MessageRole, RunStatus
from app.domain.ports import ConversationStore
from app.domain.types import JsonValue
from app.infrastructure.db.models import (
    AgentEventRow,
    AgentRunRow,
    Base,
    ConversationMetricsRow,
    ConversationRow,
    MessageRow,
    ToolCallRow,
)


class SqlAlchemyConversationStore(ConversationStore):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def create_conversation(self, title: str) -> Conversation:
        now = utc_now()
        row = ConversationRow(id=str(uuid4()), title=title, created_at=now, updated_at=now)
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        return self._to_conversation(row)

    async def list_conversations(self) -> Sequence[Conversation]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(ConversationRow).order_by(ConversationRow.updated_at.desc())
                )
            ).all()
        return [self._to_conversation(row) for row in rows]

    async def get_conversation(self, conversation_id: UUID) -> Conversation | None:
        async with self._session_factory() as session:
            row = await session.get(ConversationRow, str(conversation_id))
        return self._to_conversation(row) if row is not None else None

    async def list_messages(self, conversation_id: UUID) -> Sequence[Message]:
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(MessageRow)
                    .where(MessageRow.conversation_id == str(conversation_id))
                    .order_by(MessageRow.sequence.asc())
                )
            ).all()
        return [self._to_message(row) for row in rows]

    async def list_events(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> Sequence[AgentEvent]:
        persisted_type_values = [event_type.value for event_type in PERSISTED_AGENT_EVENT_TYPES]
        async with self._session_factory() as session:
            rows = (
                await session.scalars(
                    select(AgentEventRow)
                    .where(
                        AgentEventRow.conversation_id == str(conversation_id),
                        AgentEventRow.event_type.in_(persisted_type_values),
                    )
                    .order_by(AgentEventRow.created_at.desc(), AgentEventRow.sequence.desc())
                    .limit(limit)
                )
            ).all()
        return [self._to_event(row) for row in reversed(rows)]

    async def append_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Message:
        async with self._session_factory() as session:
            max_sequence = await session.scalar(
                select(func.max(MessageRow.sequence)).where(
                    MessageRow.conversation_id == str(conversation_id)
                )
            )
            sequence = int(max_sequence or 0) + 1
            now = utc_now()
            row = MessageRow(
                id=str(uuid4()),
                conversation_id=str(conversation_id),
                role=role.value,
                content=content,
                sequence=sequence,
                message_metadata=cast(dict[str, object], metadata or {}),
                created_at=now,
            )
            session.add(row)
            await session.execute(
                update(ConversationRow)
                .where(ConversationRow.id == str(conversation_id))
                .values(updated_at=now)
            )
            await session.commit()
        return self._to_message(row)

    async def create_run(self, conversation_id: UUID) -> AgentRun:
        row = AgentRunRow(
            id=str(uuid4()),
            conversation_id=str(conversation_id),
            status=RunStatus.RUNNING.value,
            started_at=utc_now(),
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        return self._to_run(row)

    async def finish_run(
        self,
        run_id: UUID,
        status: RunStatus,
        metrics: RunMetrics,
        error: str | None = None,
    ) -> None:
        async with self._session_factory() as session:
            await session.execute(
                update(AgentRunRow)
                .where(AgentRunRow.id == str(run_id))
                .values(
                    status=status.value,
                    input_tokens=metrics.input_tokens,
                    output_tokens=metrics.output_tokens,
                    total_tokens=metrics.total_tokens,
                    llm_calls=metrics.llm_calls,
                    tool_calls=metrics.tool_calls,
                    duration_ms=metrics.duration_ms,
                    error=error,
                    completed_at=utc_now(),
                )
            )
            await session.commit()

    async def get_conversation_metrics(
        self,
        conversation_id: UUID,
    ) -> ConversationMetrics:
        return await self.refresh_conversation_metrics(conversation_id)

    async def refresh_conversation_metrics(
        self,
        conversation_id: UUID,
    ) -> ConversationMetrics:
        async with self._session_factory() as session:
            totals = (
                await session.execute(
                    select(
                        func.coalesce(func.sum(AgentRunRow.input_tokens), 0),
                        func.coalesce(func.sum(AgentRunRow.output_tokens), 0),
                        func.coalesce(func.sum(AgentRunRow.total_tokens), 0),
                        func.coalesce(func.sum(AgentRunRow.llm_calls), 0),
                        func.coalesce(func.sum(AgentRunRow.tool_calls), 0),
                        func.coalesce(func.sum(AgentRunRow.duration_ms), 0),
                        func.count(AgentRunRow.id),
                    ).where(
                        AgentRunRow.conversation_id == str(conversation_id),
                        AgentRunRow.status.in_(
                            [RunStatus.COMPLETED.value, RunStatus.FAILED.value]
                        ),
                    )
                )
            ).one()
            row = await session.scalar(
                select(ConversationMetricsRow)
                .where(ConversationMetricsRow.conversation_id == str(conversation_id))
                .with_for_update()
            )
            if row is None:
                row = ConversationMetricsRow(
                    conversation_id=str(conversation_id),
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    llm_calls=0,
                    tool_calls=0,
                    total_duration_ms=0,
                    run_count=0,
                    updated_at=utc_now(),
                )
                session.add(row)

            row.input_tokens = int(totals[0])
            row.output_tokens = int(totals[1])
            row.total_tokens = int(totals[2])
            row.llm_calls = int(totals[3])
            row.tool_calls = int(totals[4])
            row.total_duration_ms = int(totals[5])
            row.run_count = int(totals[6])
            row.updated_at = utc_now()
            await session.commit()
            await session.refresh(row)
        return self._to_conversation_metrics(row)

    async def append_tool_call(
        self,
        *,
        run_id: UUID,
        message_id: UUID | None,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, JsonValue],
        result: JsonValue | None,
        error: str | None,
        duration_ms: int | None,
    ) -> None:
        row = ToolCallRow(
            id=str(uuid4()),
            run_id=str(run_id),
            message_id=str(message_id) if message_id else None,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=cast(dict[str, object], arguments),
            result=cast(object | None, result),
            error=error,
            duration_ms=duration_ms,
            created_at=utc_now(),
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()

    async def append_event(self, event: AgentEvent) -> None:
        row = AgentEventRow(
            id=str(event.id),
            run_id=str(event.run_id),
            conversation_id=str(event.conversation_id),
            sequence=event.sequence,
            event_type=event.type.value,
            payload=cast(dict[str, object], event.payload),
            created_at=event.timestamp,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()

    @staticmethod
    def _to_conversation(row: ConversationRow) -> Conversation:
        return Conversation(
            id=UUID(row.id),
            title=row.title,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_message(row: MessageRow) -> Message:
        return Message(
            id=UUID(row.id),
            conversation_id=UUID(row.conversation_id),
            role=MessageRole(row.role),
            content=row.content,
            sequence=row.sequence,
            created_at=row.created_at,
            metadata=cast(dict[str, JsonValue], row.message_metadata),
        )

    @staticmethod
    def _to_run(row: AgentRunRow) -> AgentRun:
        return AgentRun(
            id=UUID(row.id),
            conversation_id=UUID(row.conversation_id),
            status=RunStatus(row.status),
            metrics=RunMetrics(
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                total_tokens=row.total_tokens,
                llm_calls=row.llm_calls,
                tool_calls=row.tool_calls,
                duration_ms=row.duration_ms,
            ),
            error=row.error,
            started_at=row.started_at,
            completed_at=row.completed_at,
        )

    @staticmethod
    def _to_conversation_metrics(row: ConversationMetricsRow) -> ConversationMetrics:
        return ConversationMetrics(
            conversation_id=UUID(row.conversation_id),
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            total_tokens=row.total_tokens,
            llm_calls=row.llm_calls,
            tool_calls=row.tool_calls,
            total_duration_ms=row.total_duration_ms,
            run_count=row.run_count,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_event(row: AgentEventRow) -> AgentEvent:
        return AgentEvent(
            id=UUID(row.id),
            run_id=UUID(row.run_id),
            conversation_id=UUID(row.conversation_id),
            sequence=row.sequence,
            type=EventType(row.event_type),
            timestamp=row.created_at,
            payload=cast(dict[str, JsonValue], row.payload),
        )
