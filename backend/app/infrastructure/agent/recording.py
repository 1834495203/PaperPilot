from typing import cast
from uuid import UUID

from langchain_core.messages import AIMessage, ToolMessage

from app.application.agent import AgentRunContext
from app.domain.enums import MessageRole
from app.domain.ports import ConversationStore
from app.domain.types import JsonValue
from app.infrastructure.agent.message_mapper import text_from_message_content


class AgentExecutionRecorder:
    """Persists agent outputs without exposing storage details to graph assembly."""

    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    async def record_assistant_message(
        self,
        *,
        context: AgentRunContext,
        message: AIMessage,
        duration_ms: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> UUID:
        stored = await self._store.append_message(
            context.conversation_id,
            MessageRole.ASSISTANT,
            text_from_message_content(message.content),
            {
                "tool_calls": cast(JsonValue, message.tool_calls),
                "duration_ms": duration_ms,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            },
        )
        return stored.id

    async def record_tool_execution(
        self,
        *,
        context: AgentRunContext,
        call_id: str,
        tool_name: str,
        arguments: dict[str, JsonValue],
        content: str,
        result_summary: JsonValue | None,
        error: str | None,
        duration_ms: int,
    ) -> ToolMessage:
        stored = await self._store.append_message(
            context.conversation_id,
            MessageRole.TOOL,
            content,
            {"tool_call_id": call_id, "tool_name": tool_name},
        )
        await self._store.append_tool_call(
            run_id=context.run_id,
            message_id=stored.id,
            tool_call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result_summary,
            error=error,
            duration_ms=duration_ms,
        )
        return ToolMessage(content=content, tool_call_id=call_id, name=tool_name)
