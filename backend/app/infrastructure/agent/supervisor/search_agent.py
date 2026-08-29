import json
from time import perf_counter
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.search.tools import ArxivSearchAgentTool, ArxivSearchToolInput
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    AgentName,
    ArtifactKind,
    CompletedStep,
    DecisionSource,
)
from app.infrastructure.agent.supervisor.prompts import SEARCH_PROMPT
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import publish_metrics, with_usage


class SearchAgentNode:
    def __init__(
        self,
        *,
        model: AgentModelGateway,
        tool: ArxivSearchAgentTool,
        recorder: AgentExecutionRecorder,
    ) -> None:
        self._model = model
        self._tool = tool
        self._recorder = recorder

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        decision = state["decision"]
        if decision is None or decision.next_agent is not AgentName.SEARCH:
            raise ValueError("Search Agent requires a search decision")
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "search",
                "stage": "search",
                "summary": "Search Agent 节点开始执行",
                "objective": decision.objective,
            },
        )
        tool_call = await self._model.generate_tool_call(
            [
                SystemMessage(content=SEARCH_PROMPT),
                HumanMessage(
                    content=(
                        f"User request: {state['user_request']}\n"
                        f"Assigned objective: {decision.objective}\n"
                        f"Supervisor query hint: {decision.query or ''}"
                    )
                ),
            ],
            [self._tool.as_langchain_tool()],
        )
        if tool_call.tool_name != self._tool.name:
            raise ValueError(f"Search Agent selected unsupported tool: {tool_call.tool_name}")
        validated_call = ArxivSearchToolInput.model_validate(tool_call.arguments)
        arguments = cast(dict[str, JsonValue], validated_call.model_dump(mode="json"))
        call_id = tool_call.call_id
        query = arguments.get("query")
        query_text = query if isinstance(query, str) else decision.query or decision.objective
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "search",
                "stage": "search",
                "summary": validated_call.decision_summary,
                "query": query_text,
                "tool_name": tool_call.tool_name,
                "arguments": arguments,
            },
        )
        await context.publisher.publish(
            EventType.TOOL_STARTED.value,
            {
                "source": DecisionSource.TOOL.value,
                "actor": self._tool.name,
                "requested_by": DecisionSource.MODEL.value,
                "tool_call_id": call_id,
                "tool_name": self._tool.name,
                "arguments": arguments,
            },
        )
        started = perf_counter()
        result_summary: JsonValue | None = None
        error_message: str | None = None
        try:
            tool_result = await self._tool.execute(arguments)
            content = tool_result.content
            result_summary = tool_result.persisted_summary
            event_payload = tool_result.event_payload
            await context.publisher.publish(
                EventType.TOOL_COMPLETED.value,
                {
                    "source": DecisionSource.EXTERNAL.value,
                    "actor": self._tool.name,
                    "tool_call_id": call_id,
                    "tool_name": self._tool.name,
                    **event_payload,
                    "duration_ms": int((perf_counter() - started) * 1000),
                },
            )
        except (ValidationError, ValueError, RuntimeError) as error:
            error_message = str(error)
            content = json.dumps({"error": error_message}, ensure_ascii=False)
            await context.publisher.publish(
                EventType.TOOL_FAILED.value,
                {
                    "source": DecisionSource.EXTERNAL.value,
                    "actor": self._tool.name,
                    "tool_call_id": call_id,
                    "tool_name": self._tool.name,
                    "error": error_message,
                    "duration_ms": int((perf_counter() - started) * 1000),
                },
            )
        duration_ms = int((perf_counter() - started) * 1000)
        await self._recorder.record_tool_execution(
            context=context,
            call_id=call_id,
            tool_name=self._tool.name,
            arguments=arguments,
            content=content,
            result_summary=result_summary,
            error=error_message,
            duration_ms=duration_ms,
        )
        result_count = 0
        if isinstance(result_summary, dict):
            raw_count = result_summary.get("result_count")
            result_count = raw_count if isinstance(raw_count, int) else 0
        artifact = AgentArtifact(
            kind=ArtifactKind.SEARCH_RESULT,
            title=f"arXiv search: {query_text}",
            summary=(
                f"找到 {result_count} 篇候选论文"
                if error_message is None
                else f"arXiv 检索失败：{error_message}"
            ),
            content=content,
        )
        update: SupervisorStateUpdate = {
            "artifacts": [*state["artifacts"], artifact],
            "completed_steps": [
                *state["completed_steps"],
                CompletedStep(
                    agent=AgentName.SEARCH,
                    objective=decision.objective,
                    artifact_id=artifact.id,
                ),
            ],
            "decision": None,
            **with_usage(state, tool_call.usage, tool_calls=1),
        }
        await publish_metrics(context, update)
        return update
