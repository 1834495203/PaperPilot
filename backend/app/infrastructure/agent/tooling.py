from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from app.domain.types import JsonValue


@dataclass(frozen=True, slots=True)
class AgentToolResult:
    content: str
    result: JsonValue
    persisted_summary: JsonValue
    event_payload: dict[str, JsonValue]


class AgentTool(ABC):
    """Explicit tool boundary shared by model binding and tool execution."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def as_langchain_tool(self) -> BaseTool: ...

    @abstractmethod
    async def execute(self, arguments: dict[str, JsonValue]) -> AgentToolResult: ...
