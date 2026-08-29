from collections.abc import Sequence
from typing import cast

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)

from app.domain.entities import Message
from app.domain.enums import MessageRole


class LangChainMessageMapper:
    """Maps persisted domain messages to LangChain messages."""

    def to_langchain(self, messages: Sequence[Message]) -> list[AnyMessage]:
        result: list[AnyMessage] = []
        for message in messages:
            mapped = self._map_message(message)
            if mapped is not None:
                result.append(mapped)
        return result

    def _map_message(self, message: Message) -> AnyMessage | None:
        if message.role is MessageRole.USER:
            return HumanMessage(content=message.content)
        if message.role is MessageRole.ASSISTANT:
            tool_calls = message.metadata.get("tool_calls", [])
            return AIMessage(
                content=message.content,
                tool_calls=(
                    cast(list[ToolCall], tool_calls) if isinstance(tool_calls, list) else []
                ),
            )
        if message.role is MessageRole.TOOL:
            call_id = message.metadata.get("tool_call_id")
            tool_name = message.metadata.get("tool_name")
            if isinstance(call_id, str):
                return ToolMessage(
                    content=message.content,
                    tool_call_id=call_id,
                    name=tool_name if isinstance(tool_name, str) else None,
                )
            return None
        if message.role is MessageRole.SYSTEM:
            return SystemMessage(content=message.content)
        return None


def text_from_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            text_parts.append(cast(str, block["text"]))
    return "".join(text_parts)
