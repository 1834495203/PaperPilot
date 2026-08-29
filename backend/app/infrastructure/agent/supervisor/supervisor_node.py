from dataclasses import dataclass
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway, ModelUsage
from app.infrastructure.agent.supervisor.models import (
    AgentName,
    DecisionSource,
    SupervisorDecision,
)
from app.infrastructure.agent.supervisor.prompts import SUPERVISOR_PROMPT
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    render_artifacts,
    with_usage,
)


@dataclass(frozen=True, slots=True)
class PolicyAdjustment:
    rule: str
    summary: str
    original_value: JsonValue
    effective_value: JsonValue


@dataclass(frozen=True, slots=True)
class DecisionResolution:
    decision: SupervisorDecision
    adjustments: list[PolicyAdjustment]


class SupervisorNode:
    def __init__(self, model: AgentModelGateway, max_steps: int) -> None:
        self._model = model
        self._max_steps = max_steps

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "supervisor",
                "stage": "supervisor",
                "summary": "Supervisor 节点开始评估当前状态",
                "step": state["step_count"] + 1,
            },
        )
        if state["step_count"] >= self._max_steps:
            decision = SupervisorDecision(
                next_agent=AgentName.WRITER,
                observations=[],
                missing_information=[],
                objective="基于当前已有产物生成最终回答，并明确说明尚未解决的限制",
                decision_summary="调度步数已达到系统预算上限",
                success_criteria=["生成最终回答", "明确披露当前证据限制"],
            )
            await self._publish_policy_decision(
                context,
                decision,
                PolicyAdjustment(
                    rule="supervisor_max_steps",
                    summary="达到最大调度步数，工作流强制进入 Writer",
                    original_value=None,
                    effective_value=AgentName.WRITER.value,
                ),
            )
            return self._state_update(state, decision, None)

        completed = [step.model_dump(mode="json") for step in state["completed_steps"]]
        prompt = (
            f"User request:\n{state['user_request']}\n\n"
            f"Conversation context:\n{state['conversation_context']}\n\n"
            f"Completed steps:\n{completed}\n\n"
            f"Available artifacts:\n{render_artifacts(state['artifacts'], max_content_chars=8000)}"
        )
        result = await self._model.generate_structured(
            [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=prompt)],
            SupervisorDecision,
        )
        await self._publish_model_decision(context, result.value)
        resolution = self._normalize_decision(state, result.value)
        for adjustment in resolution.adjustments:
            await self._publish_policy_decision(context, resolution.decision, adjustment)
        update = self._state_update(state, resolution.decision, result.usage)
        await publish_metrics(context, update)
        return update

    @staticmethod
    async def _publish_model_decision(
        context: AgentRunContext,
        decision: SupervisorDecision,
    ) -> None:
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "supervisor",
                "status": "proposed",
                "stage": "supervisor",
                "summary": decision.decision_summary,
                "next_agent": decision.next_agent.value,
                "objective": decision.objective,
                "observations": cast(JsonValue, decision.observations),
                "missing_information": cast(JsonValue, decision.missing_information),
                "success_criteria": cast(JsonValue, decision.success_criteria),
                "query": decision.query,
                "artifact_ids": [str(item) for item in decision.artifact_ids],
            },
        )

    @staticmethod
    async def _publish_policy_decision(
        context: AgentRunContext,
        decision: SupervisorDecision,
        adjustment: PolicyAdjustment,
    ) -> None:
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.POLICY.value,
                "actor": "supervisor",
                "status": "effective",
                "stage": "supervisor",
                "summary": adjustment.summary,
                "policy_rule": adjustment.rule,
                "original_value": adjustment.original_value,
                "effective_value": adjustment.effective_value,
                "next_agent": decision.next_agent.value,
                "objective": decision.objective,
            },
        )

    @staticmethod
    def _state_update(
        state: SupervisorState,
        decision: SupervisorDecision,
        usage: ModelUsage | None,
    ) -> SupervisorStateUpdate:
        update: SupervisorStateUpdate = {
            "decision": decision,
            "step_count": state["step_count"] + 1,
        }
        if usage is not None:
            update.update(with_usage(state, usage))
        return update

    @staticmethod
    def _normalize_decision(
        state: SupervisorState,
        original: SupervisorDecision,
    ) -> DecisionResolution:
        decision = original
        adjustments: list[PolicyAdjustment] = []
        valid_ids = {artifact.id for artifact in state["artifacts"]}
        valid_artifact_ids = [
            artifact_id for artifact_id in decision.artifact_ids if artifact_id in valid_ids
        ]
        if valid_artifact_ids != decision.artifact_ids:
            adjustments.append(
                PolicyAdjustment(
                    rule="known_artifact_ids_only",
                    summary="移除了当前状态中不存在的 Artifact 引用",
                    original_value=[str(item) for item in decision.artifact_ids],
                    effective_value=[str(item) for item in valid_artifact_ids],
                )
            )
            decision = decision.model_copy(update={"artifact_ids": valid_artifact_ids})
        if decision.next_agent is AgentName.SEARCH and not decision.query:
            adjustments.append(
                PolicyAdjustment(
                    rule="search_query_required",
                    summary="Search 决策缺少 query，工作流使用 objective 作为检索提示",
                    original_value=None,
                    effective_value=decision.objective,
                )
            )
            decision = decision.model_copy(update={"query": decision.objective})
        if decision.next_agent is AgentName.ANALYST and not state["artifacts"]:
            adjustments.append(
                PolicyAdjustment(
                    rule="analyst_requires_artifacts",
                    summary="Analyst 没有输入 Artifact，工作流将下一步改为 Search",
                    original_value=AgentName.ANALYST.value,
                    effective_value=AgentName.SEARCH.value,
                )
            )
            decision = decision.model_copy(
                update={
                    "next_agent": AgentName.SEARCH,
                    "objective": "先检索与用户目标相关的论文证据",
                    "query": decision.query or state["user_request"][:300],
                    "artifact_ids": [],
                }
            )
        return DecisionResolution(decision=decision, adjustments=adjustments)
