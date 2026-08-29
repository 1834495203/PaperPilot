from time import perf_counter

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.domain.enums import EventType, MessageRole
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway
from app.infrastructure.agent.supervisor.models import (
    AgentName,
    CompletedStep,
    DecisionSource,
)
from app.infrastructure.agent.supervisor.prompts import WRITER_PROMPT
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    render_artifacts,
    select_artifacts,
    with_usage,
)


class WriterAgentNode:
    def __init__(
        self,
        *,
        model: AgentModelGateway,
        recorder: AgentExecutionRecorder,
    ) -> None:
        self._model = model
        self._recorder = recorder

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        decision = state["decision"]
        if decision is None or decision.next_agent is not AgentName.WRITER:
            raise ValueError("Writer Agent requires a writer decision")
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "writer",
                "stage": "writer",
                "summary": "Writer Agent 节点开始执行",
                "objective": decision.objective,
            },
        )
        selected = select_artifacts(state["artifacts"], decision.artifact_ids)
        prompt = (
            f"User request:\n{state['user_request']}\n\n"
            f"Conversation context:\n{state['conversation_context']}\n\n"
            f"Writing objective:\n{decision.objective}\n\n"
            f"Available research artifacts:\n"
            f"{render_artifacts(selected, max_content_chars=40_000)}"
        )

        async def publish_token(text: str) -> None:
            await context.publisher.publish(
                EventType.TOKEN.value,
                {
                    "source": DecisionSource.MODEL.value,
                    "actor": "writer",
                    "text": text,
                    "stage": "writer",
                },
            )

        started = perf_counter()
        result = await self._model.generate_text(
            [SystemMessage(content=WRITER_PROMPT), HumanMessage(content=prompt)],
            token_consumer=publish_token,
        )
        if not result.text.strip():
            raise RuntimeError("Writer Agent returned an empty response")
        message_id = await self._recorder.record_assistant_message(
            context=context,
            message=result.message,
            duration_ms=int((perf_counter() - started) * 1000),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        )
        await context.publisher.publish(
            EventType.MESSAGE_COMPLETED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "writer",
                "message_id": str(message_id),
                "role": MessageRole.ASSISTANT.value,
                "content": result.text,
                "has_tool_calls": False,
                "stage": "writer",
            },
        )
        update: SupervisorStateUpdate = {
            "completed_steps": [
                *state["completed_steps"],
                CompletedStep(
                    agent=AgentName.WRITER,
                    objective=decision.objective,
                ),
            ],
            **with_usage(state, result.usage),
        }
        await publish_metrics(context, update)
        return update
