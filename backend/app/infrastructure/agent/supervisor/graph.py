from collections.abc import Sequence

from langgraph.graph.state import CompiledStateGraph

from app.application.agent import AgentRunContext, AgentRunner
from app.domain.entities import Message, RunMetrics
from app.domain.enums import MessageRole
from app.infrastructure.agent.supervisor.state import SupervisorState


class SupervisorAgentGraph(AgentRunner):
    """The single application entry point for the synchronous multi-agent workflow."""

    def __init__(
        self,
        graph: CompiledStateGraph[
            SupervisorState,
            AgentRunContext,
            SupervisorState,
            SupervisorState,
        ],
    ) -> None:
        self._graph = graph

    async def run(
        self,
        *,
        history: Sequence[Message],
        context: AgentRunContext,
    ) -> RunMetrics:
        user_request = self._latest_user_request(history)
        initial_state = SupervisorState(
            user_request=user_request,
            conversation_context=self._conversation_context(history),
            research_plan=None,
            artifacts=[],
            completed_steps=[],
            decision=None,
            reader_outcome=None,
            step_count=0,
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

    @staticmethod
    def _latest_user_request(history: Sequence[Message]) -> str:
        for message in reversed(history):
            if message.role is MessageRole.USER:
                return message.content
        raise ValueError("Agent run requires a user message")

    @staticmethod
    def _conversation_context(history: Sequence[Message]) -> str:
        visible = [
            message
            for message in history
            if message.role in {MessageRole.USER, MessageRole.ASSISTANT}
        ][-12:]
        return "\n".join(
            f"{message.role.value}: {message.content}" for message in visible
        )
