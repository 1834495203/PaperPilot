from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    AgentName,
    ArtifactKind,
    CompletedStep,
    DecisionSource,
    ReadingReport,
)
from app.infrastructure.agent.supervisor.prompts import READER_PROMPT
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    render_artifacts,
    select_artifacts,
    with_usage,
)


class ReaderAgentNode:
    def __init__(self, model: AgentModelGateway) -> None:
        self._model = model

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        decision = state["decision"]
        if decision is None or decision.next_agent is not AgentName.READER:
            raise ValueError("Reader Agent requires a reader decision")
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "reader",
                "stage": "reader",
                "summary": "Reader Agent 节点开始执行",
                "objective": decision.objective,
            },
        )
        selected = select_artifacts(state["artifacts"], decision.artifact_ids)
        prompt = (
            f"User request:\n{state['user_request']}\n\n"
            f"Reading objective:\n{decision.objective}\n\n"
            f"Available material artifacts:\n{render_artifacts(selected)}\n\n"
            "If no artifact is supplied, treat the user request itself as the provided material."
        )
        result = await self._model.generate_structured(
            [SystemMessage(content=READER_PROMPT), HumanMessage(content=prompt)],
            ReadingReport,
        )
        report = result.value
        artifact = AgentArtifact(
            kind=ArtifactKind.READING_REPORT,
            title=f"Reading report: {report.paper_or_material}",
            summary=(
                f"领域：{report.domain}；研究问题：{report.research_problem}; "
                f"证据范围：{report.evidence_scope}"
            ),
            content=report.model_dump_json(),
            source_artifact_ids=[item.id for item in selected],
        )
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "reader",
                "stage": "reader",
                "summary": report.analysis_summary,
                "artifact_id": str(artifact.id),
                "source_count": len(selected),
                "evidence_count": len(report.evidence),
                "domain": report.domain,
                "research_problem": report.research_problem,
                "evidence_scope": report.evidence_scope,
                "limitations": cast(JsonValue, report.limitations),
            },
        )
        update: SupervisorStateUpdate = {
            "artifacts": [*state["artifacts"], artifact],
            "completed_steps": [
                *state["completed_steps"],
                CompletedStep(
                    agent=AgentName.READER,
                    objective=decision.objective,
                    artifact_id=artifact.id,
                ),
            ],
            "decision": None,
            **with_usage(state, result.usage),
        }
        await publish_metrics(context, update)
        return update
