import json
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.domain.entities import Message, RunMetrics
from app.domain.enums import EventType, MessageRole
from app.domain.papers import ArxivSearchInput
from app.domain.ports import ConversationStore, EventPublisher, PaperSearchGateway
from app.domain.types import JsonValue
from app.infrastructure.agent.state import AgentState, AgentStateUpdate

SYSTEM_PROMPT = """You are PaperPilot, a careful academic-paper research assistant.
Use the arXiv search tool whenever the user asks for papers, literature, prior work,
or current research. Never invent papers. Base paper claims only on tool results and
include title, authors, year, and arXiv URL.
If the query is ambiguous, choose a focused scholarly search query and explain that choice briefly.
Answer in the user's language. Keep operational summaries concise; do not reveal
private chain-of-thought.
"""


@dataclass(frozen=True, slots=True)
class GraphRunContext:
    conversation_id: UUID
    run_id: UUID
    publisher: EventPublisher


class PaperAgentGraph:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        paper_search: PaperSearchGateway,
        store: ConversationStore,
        max_tool_iterations: int,
    ) -> None:
        self._model = model
        self._paper_search = paper_search
        self._store = store
        self._max_tool_iterations = max_tool_iterations
        self._arxiv_tool = self._build_arxiv_tool()
        self._model_with_tools = model.bind_tools([self._arxiv_tool])
        self._graph = self._build_graph()

    async def run(
        self,
        *,
        history: Sequence[Message],
        context: GraphRunContext,
    ) -> RunMetrics:
        initial_state = AgentState(
            messages=self._to_langchain_messages(history),
            tool_iterations=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            llm_calls=0,
            tool_calls=0,
        )
        result = await self._graph.ainvoke(initial_state, context=context)
        return RunMetrics(
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_tokens=result["total_tokens"],
            llm_calls=result["llm_calls"],
            tool_calls=result["tool_calls"],
        )

    def _build_graph(
        self,
    ) -> CompiledStateGraph[AgentState, GraphRunContext, AgentState, AgentState]:
        builder = StateGraph(AgentState, context_schema=GraphRunContext)
        builder.add_node("agent", self._call_model)
        builder.add_node("tools", self._call_tools)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "end": END},
        )
        builder.add_edge("tools", "agent")
        return builder.compile()

    def _build_arxiv_tool(self) -> BaseTool:
        async def search_arxiv(
            query: str,
            max_results: int = 5,
            sort_by: str = "relevance",
        ) -> list[dict[str, JsonValue]]:
            """Search arXiv for academic papers and return structured metadata and abstracts."""
            search_input = ArxivSearchInput(
                query=query,
                max_results=max_results,
                sort_by=sort_by,
            )
            papers = await self._paper_search.search(search_input)
            return [
                cast(dict[str, JsonValue], paper.model_dump(mode="json")) for paper in papers
            ]

        return StructuredTool.from_function(
            coroutine=search_arxiv,
            name="search_arxiv",
            description=(
                "Search arXiv for real academic papers relevant to a focused research query."
            ),
            args_schema=ArxivSearchInput,
        )

    async def _call_model(
        self,
        state: AgentState,
        runtime: Runtime[GraphRunContext],
    ) -> AgentStateUpdate:
        context = runtime.context
        iteration = state["tool_iterations"]
        allow_tools = iteration < self._max_tool_iterations
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "stage": "agent",
                "summary": "分析对话上下文并决定是否调用文献检索工具",
                "iteration": iteration + 1,
            },
        )
        started = perf_counter()
        model = self._model_with_tools if allow_tools else self._model
        prompt: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        if not allow_tools:
            prompt.append(
                SystemMessage(content="Tool budget exhausted. Answer using existing results.")
            )

        complete_chunk: AIMessageChunk | None = None
        async for chunk in model.astream(prompt):
            message_chunk = cast(AIMessageChunk, chunk)
            complete_chunk = (
                message_chunk if complete_chunk is None else complete_chunk + message_chunk
            )
            text = self._text_from_content(message_chunk.content)
            if text:
                await context.publisher.publish(
                    EventType.TOKEN.value,
                    {"text": text, "stage": "agent"},
                )

        if complete_chunk is None:
            raise RuntimeError("The language model returned no response")

        message = AIMessage(
            content=complete_chunk.content,
            additional_kwargs=complete_chunk.additional_kwargs,
            response_metadata=complete_chunk.response_metadata,
            tool_calls=complete_chunk.tool_calls,
            usage_metadata=complete_chunk.usage_metadata,
        )
        usage: UsageMetadata = complete_chunk.usage_metadata or UsageMetadata(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens))
        metadata: dict[str, JsonValue] = {
            "tool_calls": cast(JsonValue, message.tool_calls),
            "duration_ms": int((perf_counter() - started) * 1000),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
        }
        stored_message = await self._store.append_message(
            context.conversation_id,
            MessageRole.ASSISTANT,
            self._text_from_content(message.content),
            metadata,
        )
        await context.publisher.publish(
            EventType.MESSAGE_COMPLETED.value,
            {
                "message_id": str(stored_message.id),
                "role": MessageRole.ASSISTANT.value,
                "content": self._text_from_content(message.content),
                "has_tool_calls": bool(message.tool_calls),
            },
        )
        await context.publisher.publish(
            EventType.METRICS_UPDATED.value,
            {
                "input_tokens": state["input_tokens"] + input_tokens,
                "output_tokens": state["output_tokens"] + output_tokens,
                "total_tokens": state["total_tokens"] + total_tokens,
                "llm_calls": state["llm_calls"] + 1,
                "tool_calls": state["tool_calls"],
            },
        )
        return {
            "messages": [message],
            "input_tokens": state["input_tokens"] + input_tokens,
            "output_tokens": state["output_tokens"] + output_tokens,
            "total_tokens": state["total_tokens"] + total_tokens,
            "llm_calls": state["llm_calls"] + 1,
        }

    async def _call_tools(
        self,
        state: AgentState,
        runtime: Runtime[GraphRunContext],
    ) -> AgentStateUpdate:
        context = runtime.context
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            raise TypeError("Tool node requires an AIMessage")

        tool_messages: list[AnyMessage] = []
        for call in last_message.tool_calls:
            call_id = str(call["id"])
            name = str(call["name"])
            raw_args = call["args"]
            if not isinstance(raw_args, dict):
                raise TypeError("Tool arguments must be an object")
            arguments = cast(dict[str, JsonValue], raw_args)
            started = perf_counter()
            await context.publisher.publish(
                EventType.TOOL_STARTED.value,
                {"tool_call_id": call_id, "tool_name": name, "arguments": arguments},
            )

            result: JsonValue | None = None
            error_message: str | None = None
            try:
                if name != self._arxiv_tool.name:
                    raise ValueError(f"Unknown tool: {name}")
                validated = ArxivSearchInput.model_validate(arguments)
                papers = await self._paper_search.search(validated)
                result = cast(JsonValue, [paper.model_dump(mode="json") for paper in papers])
                content = json.dumps(result, ensure_ascii=False)
                await context.publisher.publish(
                    EventType.TOOL_COMPLETED.value,
                    {
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "result_count": len(papers),
                        "papers": result,
                        "duration_ms": int((perf_counter() - started) * 1000),
                    },
                )
            except (ValidationError, ValueError, RuntimeError) as error:
                error_message = str(error)
                content = json.dumps({"error": error_message}, ensure_ascii=False)
                await context.publisher.publish(
                    EventType.TOOL_FAILED.value,
                    {
                        "tool_call_id": call_id,
                        "tool_name": name,
                        "error": error_message,
                        "duration_ms": int((perf_counter() - started) * 1000),
                    },
                )

            duration_ms = int((perf_counter() - started) * 1000)
            tool_message = await self._store.append_message(
                context.conversation_id,
                MessageRole.TOOL,
                content,
                {"tool_call_id": call_id, "tool_name": name},
            )
            await self._store.append_tool_call(
                run_id=context.run_id,
                message_id=tool_message.id,
                tool_call_id=call_id,
                tool_name=name,
                arguments=arguments,
                result=result,
                error=error_message,
                duration_ms=duration_ms,
            )
            tool_messages.append(
                ToolMessage(content=content, tool_call_id=call_id, name=name)
            )

        return {
            "messages": tool_messages,
            "tool_iterations": state["tool_iterations"] + 1,
            "tool_calls": state["tool_calls"] + len(tool_messages),
        }

    @staticmethod
    def _route_after_agent(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    @staticmethod
    def _to_langchain_messages(messages: Sequence[Message]) -> list[AnyMessage]:
        result: list[AnyMessage] = []
        for message in messages:
            if message.role is MessageRole.USER:
                result.append(HumanMessage(content=message.content))
            elif message.role is MessageRole.ASSISTANT:
                tool_calls = message.metadata.get("tool_calls", [])
                result.append(
                    AIMessage(
                        content=message.content,
                        tool_calls=(
                            cast(list[ToolCall], tool_calls)
                            if isinstance(tool_calls, list)
                            else []
                        ),
                    )
                )
            elif message.role is MessageRole.TOOL:
                call_id = message.metadata.get("tool_call_id")
                tool_name = message.metadata.get("tool_name")
                if isinstance(call_id, str):
                    result.append(
                        ToolMessage(
                            content=message.content,
                            tool_call_id=call_id,
                            name=tool_name if isinstance(tool_name, str) else None,
                        )
                    )
            elif message.role is MessageRole.SYSTEM:
                result.append(SystemMessage(content=message.content))
        return result

    @staticmethod
    def _text_from_content(content: object) -> str:
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
