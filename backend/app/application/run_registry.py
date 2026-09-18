"""Run execution registry: one buffered, fan-out stream per active run.

A run must not depend on a browser staying connected. The registry owns the task,
keeps a bounded replay buffer of every event it produced, and lets any number of
subscribers attach, detach and re-attach. Closing a stream therefore stops
*watching* a run, never the run itself; only an explicit cancel does that.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from uuid import UUID

from app.domain.entities import AgentEvent
from app.domain.enums import TERMINAL_EVENT_TYPES, EventType
from app.domain.types import JsonValue

__all__ = ["ActiveRun", "RunRegistry", "RunSubscription", "resumed_event_payload"]


@dataclass(slots=True, eq=False)
class RunSubscription:
    """One attached reader of an active run.

    Identity equality keeps subscriptions hashable so they can live in a set.
    """

    queue: asyncio.Queue[AgentEvent | None] = field(
        default_factory=lambda: asyncio.Queue()
    )


class ActiveRun:
    """A running agent task plus the events it has produced so far."""

    def __init__(
        self,
        *,
        run_id: UUID,
        conversation_id: UUID,
        task: asyncio.Task[None],
        buffer_size: int,
    ) -> None:
        self.run_id = run_id
        self.conversation_id = conversation_id
        self.task = task
        self._events: deque[AgentEvent] = deque(maxlen=buffer_size)
        self._dropped_through = 0
        self._subscribers: set[RunSubscription] = set()
        self._terminal: AgentEvent | None = None
        self.finished = asyncio.Event()

    @property
    def is_finished(self) -> bool:
        return self._terminal is not None

    @property
    def terminal_event(self) -> AgentEvent | None:
        return self._terminal

    @property
    def dropped_through_sequence(self) -> int:
        """Highest sequence trimmed from the buffer; a resume before it has a gap."""

        return self._dropped_through

    def publish(self, event: AgentEvent) -> None:
        self._events.append(event)
        if len(self._events) == self._events.maxlen and self._events:
            self._dropped_through = max(
                self._dropped_through, event.sequence - self._events.maxlen
            )
        if event.type in TERMINAL_EVENT_TYPES:
            self._terminal = event
        for subscription in list(self._subscribers):
            subscription.queue.put_nowait(event)
        if event.type in TERMINAL_EVENT_TYPES:
            self.finished.set()
            for subscription in list(self._subscribers):
                subscription.queue.put_nowait(None)

    def snapshot(self, after_sequence: int) -> tuple[list[AgentEvent], bool]:
        """Buffered events after a sequence, plus whether anything was trimmed."""

        gap = after_sequence < self._dropped_through
        return (
            [event for event in self._events if event.sequence > after_sequence],
            gap,
        )

    def attach(self) -> RunSubscription:
        subscription = RunSubscription()
        self._subscribers.add(subscription)
        return subscription

    def detach(self, subscription: RunSubscription) -> None:
        self._subscribers.discard(subscription)


class RunRegistry:
    """Tracks active runs by id and by conversation."""

    def __init__(
        self,
        *,
        buffer_size: int = 2_000,
        retention_seconds: float = 600.0,
    ) -> None:
        if buffer_size < 100:
            raise ValueError("buffer_size must be at least 100 events")
        if retention_seconds < 0:
            raise ValueError("retention_seconds cannot be negative")
        self._buffer_size = buffer_size
        self._retention_seconds = retention_seconds
        self._runs: dict[UUID, ActiveRun] = {}

    def register(
        self,
        *,
        run_id: UUID,
        conversation_id: UUID,
        task: asyncio.Task[None],
    ) -> ActiveRun:
        active = ActiveRun(
            run_id=run_id,
            conversation_id=conversation_id,
            task=task,
            buffer_size=self._buffer_size,
        )
        self._runs[run_id] = active
        return active

    def get(self, run_id: UUID) -> ActiveRun | None:
        return self._runs.get(run_id)

    def publish(self, run_id: UUID, event: AgentEvent) -> None:
        active = self._runs.get(run_id)
        if active is None:
            return
        active.publish(event)
        if active.is_finished:
            self._schedule_discard(run_id)

    def _schedule_discard(self, run_id: UUID) -> None:
        """Drop a finished run's buffer after a reconnect window.

        Retaining every finished run would leak its token buffer for the life of the
        process; after the window, a resume falls back to persisted events.
        """

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.call_later(self._retention_seconds, self.discard, run_id)

    def active_run_id(self, conversation_id: UUID) -> UUID | None:
        for run_id, active in self._runs.items():
            if active.conversation_id == conversation_id and not active.task.done():
                return run_id
        return None

    async def cancel(self, run_id: UUID) -> bool:
        active = self._runs.get(run_id)
        if active is None or active.task.done():
            return False
        active.task.cancel()
        try:
            await active.task
        except asyncio.CancelledError:
            pass
        return True

    async def wait(self, run_id: UUID) -> None:
        active = self._runs.get(run_id)
        if active is None:
            return
        try:
            await active.task
        except asyncio.CancelledError:
            pass

    def discard(self, run_id: UUID) -> None:
        """Forget a run once nothing needs to replay it."""

        active = self._runs.pop(run_id, None)
        if active is None:
            return
        for subscription in list(active._subscribers):  # noqa: SLF001
            subscription.queue.put_nowait(None)


def resumed_event_payload(
    *,
    run_id: UUID,
    replayed: int,
    gap: bool,
    active: bool,
    latest_sequence: int,
) -> dict[str, JsonValue]:
    return {
        "run_id": str(run_id),
        "replayed": replayed,
        "gap": gap,
        "active": active,
        "latest_sequence": latest_sequence,
        "summary": (
            "重新连接到正在执行的任务" if active else "任务已结束，正在回放已持久化的事件"
        ),
    }


RESUMED_EVENT_TYPE = EventType.RUN_RESUMED
