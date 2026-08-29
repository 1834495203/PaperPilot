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
    AnalysisReport,
    ArtifactKind,
    CompletedStep,
    DecisionSource,
)
from app.infrastructure.agent.supervisor.prompts import ANALYST_PROMPT
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    render_artifacts,
    select_artifacts,
    with_usage,
)


class AnalystAgentNode:
    def __init__(self, model: AgentModelGateway) -> None:
        self._model = model

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        decision = state["decision"]
        if decision is None or decision.next_agent is not AgentName.ANALYST:
            raise ValueError("Analyst Agent requires an analyst decision")
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "analyst",
                "stage": "analyst",
                "summary": "Analyst Agent 节点开始执行",
                "objective": decision.objective,
            },
        )
        selected = select_artifacts(state["artifacts"], decision.artifact_ids)
        result = await self._model.generate_structured(
            [
                SystemMessage(content=ANALYST_PROMPT),
                HumanMessage(
                    content=(
                        f"User request:\n{state['user_request']}\n\n"
                        f"Analysis objective:\n{decision.objective}\n\n"
                        "Research artifacts:\n"
                        f"{render_artifacts(selected, max_content_chars=36_000)}"
                    )
                ),
            ],
            AnalysisReport,
        )
        report = result.value
        artifact = AgentArtifact(
            kind=ArtifactKind.ANALYSIS_REPORT,
            title=f"Analysis report: {report.analysis_type}",
            summary=(
                f"形成 {len(report.findings)} 条高层发现和 "
                f"{len(report.research_gaps)} 个潜在研究空白"
            ),
            content=report.model_dump_json(),
            source_artifact_ids=[item.id for item in selected],
        )
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "analyst",
                "stage": "analyst",
                "summary": report.analysis_summary,
                "artifact_id": str(artifact.id),
                "finding_count": len(report.findings),
                "research_gap_count": len(report.research_gaps),
                "novelty_assessment": report.novelty_assessment,
                "research_gaps": cast(JsonValue, report.research_gaps),
                "unresolved_questions": cast(JsonValue, report.unresolved_questions),
            },
        )
        update: SupervisorStateUpdate = {
            "artifacts": [*state["artifacts"], artifact],
            "completed_steps": [
                *state["completed_steps"],
                CompletedStep(
                    agent=AgentName.ANALYST,
                    objective=decision.objective,
                    artifact_id=artifact.id,
                ),
            ],
            "decision": None,
            **with_usage(state, result.usage),
        }
        await publish_metrics(context, update)
        return update
