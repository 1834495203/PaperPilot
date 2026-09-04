import json
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from app.infrastructure.agent.message_mapper import text_from_message_content

ModelT = TypeVar("ModelT", bound=BaseModel)
TokenConsumer = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class StructuredModelResult[ModelT: BaseModel]:
    value: ModelT
    usage: ModelUsage


@dataclass(frozen=True, slots=True)
class TextModelResult:
    text: str
    usage: ModelUsage
    message: AIMessage


@dataclass(frozen=True, slots=True)
class ToolCallModelResult:
    call_id: str
    tool_name: str
    arguments: dict[str, object]
    usage: ModelUsage


class AgentModelGateway(ABC):
    @abstractmethod
    async def generate_structured(
        self,
        messages: Sequence[BaseMessage],
        schema: type[ModelT],
    ) -> StructuredModelResult[ModelT]: ...

    @abstractmethod
    async def generate_text(
        self,
        messages: Sequence[BaseMessage],
        token_consumer: TokenConsumer | None = None,
    ) -> TextModelResult: ...

    @abstractmethod
    async def generate_tool_call(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
    ) -> ToolCallModelResult: ...


class ChatModelGateway(AgentModelGateway):
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def generate_structured(
        self,
        messages: Sequence[BaseMessage],
        schema: type[ModelT],
    ) -> StructuredModelResult[ModelT]:
        schema_instruction = (
            "Return exactly one JSON object matching this JSON Schema. "
            "Do not use markdown fences or add commentary.\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
        augmented = [*messages]
        if not augmented:
            raise ValueError("Structured generation requires at least one message")
        last = augmented[-1]
        augmented[-1] = last.model_copy(
            update={"content": f"{text_from_message_content(last.content)}\n\n{schema_instruction}"}
        )
        structured_model = self._model.bind(response_format={"type": "json_object"})
        total_usage = ModelUsage()
        last_error: ValueError | ValidationError | None = None
        for attempt in range(2):
            response = await structured_model.ainvoke(augmented)
            if not isinstance(response, AIMessage):
                raise TypeError("Chat model must return an AIMessage")
            usage = self._usage_from_message(response)
            total_usage = ModelUsage(
                input_tokens=total_usage.input_tokens + usage.input_tokens,
                output_tokens=total_usage.output_tokens + usage.output_tokens,
                total_tokens=total_usage.total_tokens + usage.total_tokens,
            )
            content = text_from_message_content(response.content)
            try:
                value = schema.model_validate_json(self._extract_json_object(content))
                return StructuredModelResult(value=value, usage=total_usage)
            except (ValidationError, ValueError) as error:
                last_error = error
                if attempt == 0:
                    augmented.extend(
                        [
                            response,
                            HumanMessage(
                                content=(
                                    f"The previous response is invalid for {schema.__name__}. "
                                    "Rebuild the entire object from scratch and return JSON only. "
                                    "Ensure every required field is present and all strings are "
                                    "escaped, "
                                    "and every comma, bracket, and brace is valid.\n"
                                    f"Validation errors:\n{self._validation_feedback(error)}"
                                )
                            ),
                        ]
                    )
        raise ValueError(
            f"Model returned invalid {schema.__name__} after one repair attempt: {last_error}"
        )

    async def generate_text(
        self,
        messages: Sequence[BaseMessage],
        token_consumer: TokenConsumer | None = None,
    ) -> TextModelResult:
        complete_chunk: AIMessageChunk | None = None
        async for chunk in self._model.astream(messages):
            complete_chunk = chunk if complete_chunk is None else complete_chunk + chunk
            text = text_from_message_content(chunk.content)
            if text and token_consumer is not None:
                await token_consumer(text)
        if complete_chunk is None:
            raise RuntimeError("The language model returned no response")
        message = AIMessage(
            content=complete_chunk.content,
            additional_kwargs=complete_chunk.additional_kwargs,
            response_metadata=complete_chunk.response_metadata,
            tool_calls=complete_chunk.tool_calls,
            usage_metadata=complete_chunk.usage_metadata,
        )
        return TextModelResult(
            text=text_from_message_content(message.content),
            usage=self._usage_from_message(message),
            message=message,
        )

    async def generate_tool_call(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
    ) -> ToolCallModelResult:
        if not tools:
            raise ValueError("Tool-call generation requires at least one tool")
        available_tools = {tool.name: tool for tool in tools}
        augmented = list(messages)
        bound_model = self._model.bind_tools(list(tools))
        total_usage = ModelUsage()
        last_error: ValueError | ValidationError | TypeError | None = None
        for attempt in range(2):
            response = await bound_model.ainvoke(augmented)
            if not isinstance(response, AIMessage):
                raise TypeError("Chat model must return an AIMessage")
            total_usage = self._add_usage(total_usage, self._usage_from_message(response))
            try:
                call_id, tool_name, arguments = self._validated_tool_call(
                    response,
                    available_tools,
                )
                return ToolCallModelResult(
                    call_id=call_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    usage=total_usage,
                )
            except (ValidationError, ValueError, TypeError) as error:
                last_error = error
                if attempt == 0:
                    augmented.extend(
                        [
                            response,
                            HumanMessage(
                                content=(
                                    "The previous tool call is invalid. Generate exactly one new "
                                    "tool call from scratch and do not answer in plain text.\n"
                                    f"Validation errors:\n{self._validation_feedback(error)}"
                                )
                            ),
                        ]
                    )
        raise ValueError(f"Model returned an invalid tool call after one repair: {last_error}")

    @staticmethod
    def _validated_tool_call(
        response: AIMessage,
        available_tools: dict[str, BaseTool],
    ) -> tuple[str, str, dict[str, object]]:
        if len(response.tool_calls) != 1:
            invalid_details = ", ".join(
                str(call.get("error") or "unparseable arguments")
                for call in response.invalid_tool_calls
            )
            detail = f" Invalid calls: {invalid_details}." if invalid_details else ""
            raise ValueError(
                f"Expected exactly one tool call; got {len(response.tool_calls)}.{detail}"
            )
        call = response.tool_calls[0]
        tool_name = str(call["name"])
        tool = available_tools.get(tool_name)
        if tool is None:
            expected = ", ".join(sorted(available_tools))
            raise ValueError(f"Unsupported tool '{tool_name}'; expected one of: {expected}")
        raw_arguments = call["args"]
        if not isinstance(raw_arguments, dict):
            raise TypeError("Tool-call arguments must be a JSON object")
        input_schema = cast(type[BaseModel], tool.get_input_schema())
        arguments = input_schema.model_validate(raw_arguments).model_dump()
        return str(call["id"]), tool_name, cast(dict[str, object], arguments)

    @staticmethod
    def _extract_json_object(content: str) -> str:
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Model response does not contain a JSON object")
        return stripped[start : end + 1]

    @staticmethod
    def _validation_feedback(
        error: ValueError | ValidationError | TypeError,
    ) -> str:
        if isinstance(error, ValidationError):
            return json.dumps(
                error.errors(include_input=False, include_url=False),
                ensure_ascii=False,
            )
        return str(error)

    @staticmethod
    def _add_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=left.input_tokens + right.input_tokens,
            output_tokens=left.output_tokens + right.output_tokens,
            total_tokens=left.total_tokens + right.total_tokens,
        )

    @staticmethod
    def _usage_from_message(message: AIMessage) -> ModelUsage:
        usage: UsageMetadata = message.usage_metadata or UsageMetadata(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        return ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
