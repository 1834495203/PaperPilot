import json
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import cast
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.papers import Paper
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.search.tools import (
    AcademicPaperSearchAgentTool,
    AcademicPaperSearchToolInput,
)
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway, ModelUsage
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    AgentName,
    ArtifactKind,
    CompletedStep,
    DecisionSource,
    PaperAssessment,
    PaperRelevance,
    SearchAgentSummary,
    SearchFailure,
    SearchPaperSummary,
    SearchReport,
    SearchScreening,
    SearchStatus,
    SearchTask,
)
from app.infrastructure.agent.supervisor.prompts import (
    SEARCH_PROMPT,
    SEARCH_SCREENING_PROMPT,
)
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import publish_metrics, with_usage


@dataclass(frozen=True, slots=True)
class SearchExecution:
    papers: list[Paper]
    failure: SearchFailure | None


class SearchAgentNode:
    def __init__(
        self,
        *,
        model: AgentModelGateway,
        tool: AcademicPaperSearchAgentTool,
        recorder: AgentExecutionRecorder,
        max_iterations: int = 2,
    ) -> None:
        self._model = model
        self._tool = tool
        self._recorder = recorder
        self._max_iterations = max_iterations

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        decision = state["decision"]
        if decision is None or not isinstance(decision.task, SearchTask):
            raise ValueError("Search Agent requires a search decision")
        task = decision.task
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "search",
                "stage": "search",
                "summary": "Search Agent 开始检索、筛选并按缺口改写查询",
                "objective": task.objective,
            },
        )

        attempted_queries, papers_by_id, assessments_by_id = self._prior_search_state(
            state["artifacts"], task.prior_search_artifact_ids
        )
        screening_summaries: list[str] = []
        screening_feedback = "No previous screening is available."
        required_query: str | None = task.query
        total_usage = ModelUsage()
        llm_calls = 0
        tool_calls = 0
        successful_searches = 0
        failures: list[SearchFailure] = []

        for iteration in range(self._max_iterations):
            tool_call = await self._model.generate_tool_call(
                [
                    SystemMessage(content=SEARCH_PROMPT),
                    HumanMessage(
                        content=(
                            f"Assigned search objective: {task.objective}\n"
                            f"Supervisor query hint: {task.query}\n"
                            f"Previous screening: {screening_feedback}\n"
                            f"Required rewritten query: {required_query or 'none'}\n"
                            f"Current iteration: {iteration + 1}/{self._max_iterations}"
                        )
                    ),
                ],
                [self._tool.as_langchain_tool()],
            )
            total_usage = self._add_usage(total_usage, tool_call.usage)
            llm_calls += 1
            if tool_call.tool_name != self._tool.name:
                raise ValueError(f"Search Agent selected unsupported tool: {tool_call.tool_name}")
            validated_call = AcademicPaperSearchToolInput.model_validate(tool_call.arguments)
            arguments = cast(dict[str, JsonValue], validated_call.model_dump(mode="json"))
            query_text = validated_call.query
            if query_text in attempted_queries:
                await context.publisher.publish(
                    EventType.DECISION_RECORDED.value,
                    {
                        "source": DecisionSource.POLICY.value,
                        "actor": "search",
                        "stage": "search",
                        "summary": "改写查询与已执行查询重复，停止本次 Search 内部循环",
                        "query": query_text,
                    },
                )
                break
            attempted_queries.append(query_text)
            execution = await self._execute_tool(
                context=context,
                call_id=tool_call.call_id,
                validated_call=validated_call,
                arguments=arguments,
            )
            tool_calls += 1
            if execution.failure is not None:
                failures.append(execution.failure)
                break
            successful_searches += 1
            for paper in execution.papers:
                papers_by_id[paper.paper_id] = paper
            if not papers_by_id:
                screening_feedback = "No papers were returned."
                required_query = None
                continue

            candidates = [paper.model_dump(mode="json") for paper in papers_by_id.values()]
            screening_result = await self._model.generate_structured(
                [
                    SystemMessage(content=SEARCH_SCREENING_PROMPT),
                    HumanMessage(
                        content=(
                            f"Search objective:\n{task.objective}\n\n"
                            f"Attempted queries:\n{attempted_queries}\n\n"
                            f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
                        )
                    ),
                ],
                SearchScreening,
            )
            total_usage = self._add_usage(total_usage, screening_result.usage)
            llm_calls += 1
            screening = self._normalize_screening(screening_result.value, papers_by_id)
            screening_summaries.append(screening.screening_summary)
            assessments_by_id.update(
                {assessment.paper_id: assessment for assessment in screening.assessments}
            )
            await context.publisher.publish(
                EventType.DECISION_RECORDED.value,
                {
                    "source": DecisionSource.MODEL.value,
                    "actor": "search",
                    "stage": "screening",
                    "summary": screening.screening_summary,
                    "direct_count": self._count_relevance(
                        assessments_by_id.values(), PaperRelevance.DIRECT
                    ),
                    "adjacent_count": self._count_relevance(
                        assessments_by_id.values(), PaperRelevance.ADJACENT
                    ),
                    "irrelevant_count": self._count_relevance(
                        assessments_by_id.values(), PaperRelevance.IRRELEVANT
                    ),
                    "continue_search": screening.continue_search,
                    "rewritten_query": screening.rewritten_query,
                },
            )
            if not screening.continue_search or not screening.rewritten_query:
                break
            screening_feedback = screening.screening_summary
            required_query = screening.rewritten_query

        accepted_ids = {
            paper_id
            for paper_id, assessment in assessments_by_id.items()
            if assessment.relevance in {PaperRelevance.DIRECT, PaperRelevance.ADJACENT}
        }
        accepted_papers = [
            paper for paper_id, paper in papers_by_id.items() if paper_id in accepted_ids
        ]
        assessments = list(assessments_by_id.values())
        direct_count = self._count_relevance(assessments, PaperRelevance.DIRECT)
        adjacent_count = self._count_relevance(assessments, PaperRelevance.ADJACENT)
        irrelevant_count = self._count_relevance(assessments, PaperRelevance.IRRELEVANT)
        status = self._search_status(
            failures=failures,
            successful_searches=successful_searches,
            accepted_papers=accepted_papers,
        )
        report = SearchReport(
            attempted_queries=attempted_queries,
            status=status,
            failures=failures,
            papers=accepted_papers,
            assessments=assessments,
            screening_summary=(
                screening_summaries[-1]
                if screening_summaries
                else (
                    failures[-1].message
                    if failures
                    else "Search returned no candidates that could be screened"
                )
            ),
        )
        artifact = AgentArtifact(
            title=f"Screened academic search: {task.objective}",
            supervisor_summary=SearchAgentSummary(
                summary=(
                    f"检索状态：{status.value}。执行 {len(attempted_queries)} 个查询；"
                    f"筛选出 {direct_count} 篇直接相关、"
                    f"{adjacent_count} 篇相邻、排除 {irrelevant_count} 篇。"
                    f"{report.screening_summary}"
                ),
                status=status,
                failures=failures,
                attempted_queries=attempted_queries,
                direct_count=direct_count,
                adjacent_count=adjacent_count,
                irrelevant_count=irrelevant_count,
                papers=[
                    SearchPaperSummary(
                        paper_id=item.paper_id,
                        source=papers_by_id[item.paper_id].source,
                        title=papers_by_id[item.paper_id].title,
                        relevance=item.relevance,
                        relevance_reason=item.relevance_reason,
                        matched_topics=item.matched_topics,
                    )
                    for item in assessments
                    if item.paper_id in papers_by_id
                ],
            ),
            content=report.model_dump_json(),
            source_artifact_ids=task.prior_search_artifact_ids,
        )
        update: SupervisorStateUpdate = {
            "artifacts": [*state["artifacts"], artifact],
            "completed_steps": [
                *state["completed_steps"],
                CompletedStep(
                    agent=AgentName.SEARCH,
                    objective=task.objective,
                    artifact_id=artifact.id,
                ),
            ],
            "decision": None,
            **with_usage(
                state,
                total_usage,
                llm_calls=llm_calls,
                tool_calls=tool_calls,
            ),
        }
        await publish_metrics(context, update)
        return update

    async def _execute_tool(
        self,
        *,
        context: AgentRunContext,
        call_id: str,
        validated_call: AcademicPaperSearchToolInput,
        arguments: dict[str, JsonValue],
    ) -> SearchExecution:
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "search",
                "stage": "search",
                "summary": validated_call.decision_summary,
                "query": validated_call.query,
                "tool_name": self._tool.name,
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
        failure: SearchFailure | None = None
        papers: list[Paper] = []
        content = "[]"
        try:
            tool_result = await self._tool.execute(arguments)
            content = tool_result.content
            result_summary = tool_result.persisted_summary
            papers = self._papers_from_result(tool_result.result)
            await context.publisher.publish(
                EventType.TOOL_COMPLETED.value,
                {
                    "source": DecisionSource.EXTERNAL.value,
                    "actor": self._tool.name,
                    "tool_call_id": call_id,
                    "tool_name": self._tool.name,
                    **tool_result.event_payload,
                    "duration_ms": int((perf_counter() - started) * 1000),
                },
            )
        except (ValidationError, ValueError, RuntimeError) as error:
            failure = self._failure_from_error(error)
            content = json.dumps({"error": failure.model_dump(mode="json")}, ensure_ascii=False)
            await context.publisher.publish(
                EventType.TOOL_FAILED.value,
                {
                    "source": DecisionSource.EXTERNAL.value,
                    "actor": self._tool.name,
                    "tool_call_id": call_id,
                    "tool_name": self._tool.name,
                    "error": failure.message,
                    "error_category": failure.category,
                    "retryable": failure.retryable,
                    "status_code": failure.status_code,
                    "retry_after_seconds": failure.retry_after_seconds,
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
            error=failure.message if failure is not None else None,
            duration_ms=duration_ms,
        )
        return SearchExecution(papers=papers, failure=failure)

    @staticmethod
    def _failure_from_error(error: Exception) -> SearchFailure:
        return SearchFailure(
            category=str(getattr(error, "category", "tool_error")),
            message=str(error) or type(error).__name__,
            retryable=bool(getattr(error, "retryable", False)),
            status_code=getattr(error, "status_code", None),
            retry_after_seconds=getattr(error, "retry_after_seconds", None),
        )

    @staticmethod
    def _search_status(
        *,
        failures: list[SearchFailure],
        successful_searches: int,
        accepted_papers: list[Paper],
    ) -> SearchStatus:
        if failures:
            category = failures[-1].category
            if category == "rate_limited":
                return SearchStatus.RATE_LIMITED
            if category in {"provider_error", "network_error"}:
                return SearchStatus.PROVIDER_ERROR
            return SearchStatus.FAILED
        if successful_searches > 0 and not accepted_papers:
            return SearchStatus.NO_MATCHES
        return SearchStatus.COMPLETED

    @staticmethod
    def _papers_from_result(result: JsonValue) -> list[Paper]:
        if not isinstance(result, list):
            raise TypeError("Academic search tool result must be a list")
        return [Paper.model_validate(item) for item in result]

    @staticmethod
    def _prior_search_state(
        artifacts: list[AgentArtifact],
        prior_ids: list[UUID],
    ) -> tuple[list[str], dict[str, Paper], dict[str, PaperAssessment]]:
        selected_ids = {str(item) for item in prior_ids}
        attempted_queries: list[str] = []
        papers_by_id: dict[str, Paper] = {}
        assessments_by_id: dict[str, PaperAssessment] = {}
        for artifact in artifacts:
            if (
                str(artifact.id) not in selected_ids
                or artifact.kind is not ArtifactKind.SEARCH_RESULT
            ):
                continue
            report = SearchReport.model_validate_json(artifact.content)
            for query in report.attempted_queries:
                if query not in attempted_queries:
                    attempted_queries.append(query)
            papers_by_id.update({paper.paper_id: paper for paper in report.papers})
            assessments_by_id.update(
                {assessment.paper_id: assessment for assessment in report.assessments}
            )
        return attempted_queries, papers_by_id, assessments_by_id

    @staticmethod
    def _normalize_screening(
        screening: SearchScreening,
        papers_by_id: dict[str, Paper],
    ) -> SearchScreening:
        known_ids = set(papers_by_id)
        normalized: dict[str, PaperAssessment] = {
            item.paper_id: item for item in screening.assessments if item.paper_id in known_ids
        }
        for missing_id in known_ids - set(normalized):
            normalized[missing_id] = PaperAssessment(
                paper_id=missing_id,
                relevance=PaperRelevance.IRRELEVANT,
                relevance_reason="The screening response omitted this candidate",
                matched_topics=[],
            )
        return screening.model_copy(update={"assessments": list(normalized.values())})

    @staticmethod
    def _count_relevance(
        assessments: Iterable[PaperAssessment],
        relevance: PaperRelevance,
    ) -> int:
        return sum(1 for item in assessments if item.relevance is relevance)

    @staticmethod
    def _add_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
        return ModelUsage(
            input_tokens=left.input_tokens + right.input_tokens,
            output_tokens=left.output_tokens + right.output_tokens,
            total_tokens=left.total_tokens + right.total_tokens,
        )
