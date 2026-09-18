from dataclasses import dataclass
from typing import cast
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway, ModelUsage
from app.infrastructure.agent.supervisor.models import (
    AgentName,
    AnalystTask,
    ArtifactKind,
    DecisionAssessment,
    DecisionSource,
    ReaderTask,
    SearchTask,
    SupervisorDecision,
    WriterTask,
)
from app.infrastructure.agent.supervisor.prompts import SUPERVISOR_PROMPT
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    render_supervisor_context,
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


@dataclass(frozen=True, slots=True)
class ReaderScopeResolution:
    """Resolved reader scope; ``task is None`` means the workflow must search first."""

    task: ReaderTask | None
    adjustments: list[PolicyAdjustment]
    recovery_rule: str | None = None
    recovery_original: str = ""


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
                "summary": "Supervisor 节点开始评估 Agent 摘要和当前目标",
                "step": state["step_count"] + 1,
            },
        )
        if state["step_count"] >= self._max_steps:
            decision = SupervisorDecision(
                assessment=DecisionAssessment(
                    observations=[],
                    missing_information=["工作流已达到调度预算上限"],
                    decision_summary="调度步数已达到系统预算上限",
                ),
                task=WriterTask(
                    objective="基于当前已有产物生成最终回答，并明确说明尚未解决的限制",
                    source_artifact_ids=[artifact.id for artifact in state["artifacts"]],
                ),
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
        research_plan = state["research_plan"]
        prompt = (
            f"User request:\n{state['user_request']}\n\n"
            f"Conversation context:\n{state['conversation_context']}\n\n"
            f"Local indexed corpus available:\n{context.local_corpus_available}\n\n"
            f"Explicitly scoped local paper IDs (normally empty):\n"
            f"{list(context.paper_ids)}\n\n"
            f"Research plan (authoritative task strategy unless completed steps "
            f"contradict it):\n"
            f"{research_plan.model_dump_json() if research_plan is not None else 'not planned'}"
            f"\n\nCompleted steps:\n{completed}\n\n"
            "Agent-authored supervisor summaries (detailed reports are available to downstream "
            f"agents by ID):\n{render_supervisor_context(state['artifacts'])}"
        )
        result = await self._model.generate_structured(
            [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=prompt)],
            SupervisorDecision,
        )
        await self._publish_model_decision(context, result.value)
        resolution = self._normalize_decision(
            state,
            result.value,
            local_paper_ids=set(context.paper_ids),
            local_corpus_available=context.local_corpus_available,
        )
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
        assessment = decision.assessment
        task = decision.task
        payload: dict[str, JsonValue] = {
            "source": DecisionSource.MODEL.value,
            "actor": "supervisor",
            "status": "proposed",
            "stage": "supervisor",
            "summary": assessment.decision_summary,
            "next_agent": task.agent.value,
            "objective": task.objective,
            "observations": cast(JsonValue, assessment.observations),
            "missing_information": cast(JsonValue, assessment.missing_information),
            **SupervisorNode._task_event_payload(task),
        }
        await context.publisher.publish(EventType.DECISION_RECORDED.value, payload)

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
                "next_agent": decision.task.agent.value,
                "objective": decision.task.objective,
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
        *,
        local_paper_ids: set[str] | None = None,
        local_corpus_available: bool = False,
    ) -> DecisionResolution:
        decision = original
        task = decision.task
        assessment = decision.assessment
        adjustments: list[PolicyAdjustment] = []
        valid_ids = {artifact.id for artifact in state["artifacts"]}
        valid_search_ids = {
            artifact.id
            for artifact in state["artifacts"]
            if artifact.kind is ArtifactKind.SEARCH_RESULT
        }
        available_local_ids = local_paper_ids or set()

        if isinstance(task, SearchTask):
            valid_prior_ids = [
                artifact_id
                for artifact_id in task.prior_search_artifact_ids
                if artifact_id in valid_search_ids
            ]
            if valid_prior_ids != task.prior_search_artifact_ids:
                adjustments.append(
                    PolicyAdjustment(
                        rule="known_search_artifact_ids_only",
                        summary="移除了不存在的历史 Search Artifact 引用",
                        original_value=[str(item) for item in task.prior_search_artifact_ids],
                        effective_value=[str(item) for item in valid_prior_ids],
                    )
                )
                task = task.model_copy(update={"prior_search_artifact_ids": valid_prior_ids})
        elif isinstance(task, ReaderTask):
            scope = SupervisorNode._normalize_reader_scope(
                task,
                available_local_ids=available_local_ids,
                local_corpus_available=local_corpus_available,
                valid_search_ids=valid_search_ids,
            )
            adjustments.extend(scope.adjustments)
            if scope.task is None:
                adjustments.append(
                    PolicyAdjustment(
                        rule=scope.recovery_rule or "reader_requires_search_source",
                        summary="Reader 必须引用有效的 Search Artifact，工作流改为补充检索",
                        original_value=scope.recovery_original,
                        effective_value=AgentName.SEARCH.value,
                    )
                )
                task = SupervisorNode._recovery_search_task(state)
            else:
                task = scope.task
        elif isinstance(task, (AnalystTask, WriterTask)):
            valid_source_ids = [
                artifact_id for artifact_id in task.source_artifact_ids if artifact_id in valid_ids
            ]
            if valid_source_ids != task.source_artifact_ids:
                adjustments.append(
                    PolicyAdjustment(
                        rule="known_source_artifact_ids_only",
                        summary="移除了不存在的下游 Artifact 引用",
                        original_value=[str(item) for item in task.source_artifact_ids],
                        effective_value=[str(item) for item in valid_source_ids],
                    )
                )
                task = task.model_copy(update={"source_artifact_ids": valid_source_ids})
            if isinstance(task, AnalystTask) and not valid_source_ids:
                adjustments.append(
                    PolicyAdjustment(
                        rule="analyst_requires_sources",
                        summary="Analyst 没有有效来源，工作流改为补充检索",
                        original_value=AgentName.ANALYST.value,
                        effective_value=AgentName.SEARCH.value,
                    )
                )
                task = SupervisorNode._recovery_search_task(state)

        decision = decision.model_copy(update={"assessment": assessment, "task": task})
        return DecisionResolution(decision=decision, adjustments=adjustments)

    @staticmethod
    def _normalize_reader_scope(
        task: ReaderTask,
        *,
        available_local_ids: set[str],
        local_corpus_available: bool,
        valid_search_ids: set[UUID],
    ) -> ReaderScopeResolution:
        """Resolve single and multi-paper reader scope against what actually exists.

        A multi-paper objective is the normal shape for a comparison, so the policy
        keeps every locally indexed target instead of collapsing it to one paper,
        and only falls back to a search when no requested paper is readable at all.
        """

        adjustments: list[PolicyAdjustment] = []
        targets = task.target_paper_ids
        if not targets:
            if local_corpus_available or available_local_ids:
                if task.source_artifact_id is not None:
                    adjustments.append(
                        PolicyAdjustment(
                            rule="global_local_reader_does_not_require_search_source",
                            summary="全库检索不需要 Search Artifact，已移除该引用",
                            original_value=str(task.source_artifact_id),
                            effective_value=None,
                        )
                    )
                    return ReaderScopeResolution(
                        task=task.model_copy(update={"source_artifact_id": None}),
                        adjustments=adjustments,
                    )
                return ReaderScopeResolution(task=task, adjustments=adjustments)
            if task.source_artifact_id in valid_search_ids:
                return ReaderScopeResolution(task=task, adjustments=adjustments)
            return ReaderScopeResolution(
                task=None,
                adjustments=adjustments,
                recovery_original=str(task.source_artifact_id),
            )

        local_targets = [
            paper_id for paper_id in targets if paper_id in available_local_ids
        ]
        if local_targets:
            dropped = [paper_id for paper_id in targets if paper_id not in local_targets]
            if not dropped and len(local_targets) == 1 and task.paper_id is not None:
                if task.source_artifact_id is None:
                    return ReaderScopeResolution(task=task, adjustments=adjustments)
                adjustments.append(
                    PolicyAdjustment(
                        rule="local_reader_does_not_require_search_source",
                        summary="本地已建库论文不需要 Search Artifact，已移除该引用",
                        original_value=str(task.source_artifact_id),
                        effective_value=None,
                    )
                )
                return ReaderScopeResolution(
                    task=task.model_copy(update={"source_artifact_id": None}),
                    adjustments=adjustments,
                )
            updates: dict[str, object] = {
                "paper_ids": local_targets,
                "paper_id": None,
                "source_artifact_id": None,
            }
            if dropped:
                adjustments.append(
                    PolicyAdjustment(
                        rule="reader_local_targets_only",
                        summary="只保留本地已建库论文；外部论文需要单独的 Search Artifact",
                        original_value=[*targets],
                        effective_value=[*local_targets],
                    )
                )
            elif task.source_artifact_id is not None:
                adjustments.append(
                    PolicyAdjustment(
                        rule="local_reader_does_not_require_search_source",
                        summary="本地已建库论文不需要 Search Artifact，已移除该引用",
                        original_value=str(task.source_artifact_id),
                        effective_value=None,
                    )
                )
            return ReaderScopeResolution(
                task=task.model_copy(update=updates),
                adjustments=adjustments,
            )

        if task.source_artifact_id not in valid_search_ids:
            return ReaderScopeResolution(
                task=None,
                adjustments=adjustments,
                recovery_original=",".join(targets),
            )
        if len(targets) == 1:
            return ReaderScopeResolution(task=task, adjustments=adjustments)
        adjustments.append(
            PolicyAdjustment(
                rule="external_reader_single_paper_only",
                summary="外部 PDF 阅读一次只支持一篇论文，已限定为第一篇",
                original_value=[*targets],
                effective_value=targets[0],
            )
        )
        return ReaderScopeResolution(
            task=task.model_copy(
                update={"paper_id": targets[0], "paper_ids": []}
            ),
            adjustments=adjustments,
        )

    @staticmethod
    def _recovery_search_task(state: SupervisorState) -> SearchTask:
        return SearchTask(
            objective="根据 Agent 摘要中尚未满足的条件补充直接相关论文证据",
            query=state["user_request"],
            prior_search_artifact_ids=[
                artifact.id
                for artifact in state["artifacts"]
                if artifact.kind is ArtifactKind.SEARCH_RESULT
            ],
        )

    @staticmethod
    def _task_event_payload(
        task: SearchTask | ReaderTask | AnalystTask | WriterTask,
    ) -> dict[str, JsonValue]:
        if isinstance(task, SearchTask):
            return {
                "query": task.query,
                "artifact_ids": cast(
                    JsonValue,
                    [str(item) for item in task.prior_search_artifact_ids],
                ),
            }
        if isinstance(task, ReaderTask):
            return {
                "artifact_ids": (
                    [str(task.source_artifact_id)]
                    if task.source_artifact_id is not None
                    else []
                ),
                "paper_ids": cast(JsonValue, task.target_paper_ids),
                "retrieval_scope": (
                    "all_local_papers"
                    if not task.target_paper_ids
                    else "explicit_papers"
                ),
                "reader_depth": task.depth.value,
            }
        return {
            "artifact_ids": cast(
                JsonValue,
                [str(item) for item in task.source_artifact_ids],
            )
        }
