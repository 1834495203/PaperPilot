import hashlib
import json
from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

import pytest
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.ports import EventPublisher
from app.domain.rag import (
    CoverageCell,
    CoverageMatrix,
    CoverageStatus,
    RetrievalHit,
    RetrievalMode,
    RetrievalSource,
    RetrievalStrategy,
    TreeRetrievalReport,
)
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.model_gateway import (
    AgentModelGateway,
    ModelUsage,
    StructuredModelResult,
)
from app.infrastructure.agent.supervisor.models import (
    CoverageGapJudgment,
    DecisionAssessment,
    DecisionSource,
    PaperEvidence,
    ReaderAgentSummary,
    ReaderDepth,
    ReaderTask,
    ReadingEvidenceAssessment,
    ReadingGapQuery,
    ReadingPlan,
    ReadingReport,
    ReadingSubQuestion,
    ResearchPlan,
    ResearchTaskType,
    SupervisorDecision,
)
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.state import SupervisorState


class CapturingPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, JsonValue]]] = []

    async def publish(self, event_type: str, payload: dict[str, JsonValue]) -> None:
        self.events.append((event_type, payload))


def evidence_id(chunk_id: str) -> str:
    return f"E-{hashlib.sha256(chunk_id.encode('utf-8')).hexdigest()[:12]}"


def _state(task: ReaderTask, *, research_plan: ResearchPlan | None = None) -> SupervisorState:
    return SupervisorState(
        user_request="Compare the indexed papers",
        conversation_context="user: Compare the indexed papers",
        research_plan=research_plan,
        artifacts=[],
        completed_steps=[],
        decision=SupervisorDecision(
            assessment=DecisionAssessment(
                observations=["Several local papers are selected"],
                missing_information=["Method, results and limitations per paper"],
                decision_summary="Compare the selected papers",
            ),
            task=task,
        ),
        reader_outcome=None,
        step_count=0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        llm_calls=0,
        tool_calls=0,
    )


def _hit(paper_id: str, dimension: str, *, index: int = 1) -> RetrievalHit:
    return RetrievalHit(
        rank=index,
        node_id=f"{paper_id}:chunk:{index}",
        paper_id=paper_id,
        paper_title=f"{paper_id} title",
        section_path=["3 Method"],
        page_start=3,
        page_end=3,
        text=f"{paper_id} evidence about {dimension}",
        vector_score=0.9,
        ranking_score=0.9,
        matched_dimensions=[dimension],
        matched_queries=[f"{dimension} of {paper_id}"],
        source=RetrievalSource.VECTOR,
    )


def _report(
    *,
    paper_ids: list[str],
    dimensions: list[str],
    hits: list[RetrievalHit],
    strategy: RetrievalStrategy = RetrievalStrategy.MULTI_PAPER,
) -> TreeRetrievalReport:
    matrix = CoverageMatrix.build(paper_ids=paper_ids, dimensions=dimensions)
    for hit in hits:
        for dimension in hit.matched_dimensions:
            cell = matrix.cell_for(hit.paper_id, dimension)
            if cell is None:
                continue
            matrix = matrix.with_cell(
                CoverageCell(
                    paper_id=cell.paper_id,
                    dimension=cell.dimension,
                    status=CoverageStatus.COVERED,
                    evidence_ids=[hit.node_id],
                )
            )
    return TreeRetrievalReport(
        query="compare",
        mode=RetrievalMode.COMPARE,
        paper_ids=paper_ids,
        strategy=strategy,
        searched_globally=False,
        candidate_paper_ids=paper_ids,
        missing_paper_ids=[],
        coverage=matrix,
        initial_hit_count=len(hits),
        expanded_candidate_count=0,
        hits=hits,
    )


def _context(publisher: CapturingPublisher, paper_ids: tuple[str, ...]) -> AgentRunContext:
    return AgentRunContext(
        conversation_id=uuid4(),
        run_id=uuid4(),
        publisher=publisher,
        paper_ids=paper_ids,
    )


def test_reader_task_accepts_multiple_papers_and_rejects_conflicting_scope() -> None:
    task = ReaderTask(
        objective="Compare the selected papers",
        depth=ReaderDepth.DEEP,
        paper_ids=["paper-a", "paper-b"],
    )

    assert task.target_paper_ids == ["paper-a", "paper-b"]

    with pytest.raises(ValueError):
        ReaderTask(
            objective="Compare the selected papers",
            depth=ReaderDepth.DEEP,
            paper_id="paper-a",
            paper_ids=["paper-b"],
        )


def test_reading_plan_expands_the_single_query_form_into_one_sub_question() -> None:
    plan = ReadingPlan(
        needs_retrieval=True,
        query="tree retrieval method",
        mode=RetrievalMode.METHOD,
        evidence_requirements=["method"],
        rationale="One focused query",
    )

    sub_questions = plan.retrieval_queries()

    assert len(sub_questions) == 1
    assert sub_questions[0].retrieval_query == "tree retrieval method"
    assert sub_questions[0].dimension is None


@pytest.mark.asyncio
async def test_reader_plans_and_executes_one_query_per_paper_and_dimension() -> None:
    papers = ["paper-a", "paper-b"]
    sub_questions = [
        ReadingSubQuestion(
            question=f"method of {paper_id}",
            retrieval_query=f"retrieval method of {paper_id}",
            mode=RetrievalMode.METHOD,
            paper_ids=[paper_id],
            dimension="method",
        )
        for paper_id in papers
    ]
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query=sub_questions[0].retrieval_query,
                mode=RetrievalMode.METHOD,
                strategy=RetrievalStrategy.MULTI_PAPER,
                coverage_dimensions=["method"],
                sub_questions=sub_questions,
                evidence_requirements=["method per paper"],
                rationale="One sub-question per paper for the method dimension",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="Both papers contributed method evidence",
                covered_requirements=["method per paper"],
                missing_requirements=[],
                retry_recommended=False,
                coverage_judgments=[
                    CoverageGapJudgment(
                        paper_id="paper-a",
                        dimension="method",
                        status="covered",
                        reason="The retrieved chunk describes the construction method",
                        evidence_ids=[evidence_id("paper-a:chunk:1")],
                    ),
                    CoverageGapJudgment(
                        paper_id="paper-b",
                        dimension="method",
                        status="covered",
                        reason="The retrieved chunk describes the construction method",
                        evidence_ids=[evidence_id("paper-b:chunk:2")],
                    ),
                ],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="Two local papers",
                analysis_summary="Both papers describe their retrieval method.",
                objective_satisfied=True,
                answered_points=["method per paper"],
                blocking_gaps=[],
                evidence=[
                    PaperEvidence(
                        evidence_id=f"E-{'0' * 11}{index}",
                        claim="Method evidence",
                        excerpt="method",
                    )
                    for index in (1, 2)
                ],
                evidence_scope="Two rounds of local retrieval",
                limitations=[],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).return_value = _report(
        paper_ids=papers,
        dimensions=["method"],
        hits=[_hit("paper-a", "method", index=1), _hit("paper-b", "method", index=2)],
        )
    publisher = CapturingPublisher()

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        _state(
            ReaderTask(
                objective="Compare the retrieval method of both papers",
                depth=ReaderDepth.DEEP,
                paper_ids=papers,
            )
        ),
        Runtime(context=_context(publisher, ("paper-a", "paper-b"))),
    )

    call = cast(AsyncMock, retriever.retrieve).await_args
    assert call.args[0] == "retrieval method of paper-a"
    # The overall scope is the union of every sub-question scope, not the first one.
    assert call.kwargs["paper_ids"] == ["paper-a", "paper-b"]
    assert call.kwargs["strategy"] is RetrievalStrategy.MULTI_PAPER
    queries = call.kwargs["queries"]
    assert [item.query for item in queries] == [
        "retrieval method of paper-a",
        "retrieval method of paper-b",
    ]
    assert [item.paper_ids for item in queries] == [["paper-a"], ["paper-b"]]
    assert [item.dimension for item in queries] == ["method", "method"]

    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.retrieval_strategy is RetrievalStrategy.MULTI_PAPER
    assert summary.target_paper_ids == papers
    assert summary.coverage is not None
    assert summary.coverage.coverage_ratio == 1.0
    assert summary.coverage.covered_cell_count == 2
    assert summary.missing_paper_ids == []
    artifact = json.loads(update["artifacts"][-1].content)
    assert artifact["coverage_matrix"]["covered_cell_count"] == 2


@pytest.mark.asyncio
async def test_reader_repairs_judged_coverage_gaps_and_records_not_stated_cells() -> None:
    papers = ["paper-a", "paper-b"]
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="limitations of both papers",
                mode=RetrievalMode.COMPARE,
                strategy=RetrievalStrategy.MULTI_PAPER,
                coverage_dimensions=["limitations"],
                sub_questions=[
                    ReadingSubQuestion(
                        question="limitations",
                        retrieval_query="limitations of both papers",
                        mode=RetrievalMode.COMPARE,
                        dimension="limitations",
                    )
                ],
                evidence_requirements=["limitations per paper"],
                rationale="Retrieve limitations for both papers",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=False,
                coverage_summary="Only paper-a limitations were retrieved",
                covered_requirements=["paper-a limitations"],
                missing_requirements=["paper-b limitations"],
                retry_recommended=True,
                next_queries=[
                    ReadingGapQuery(
                        query="paper-b failure cases and future work",
                        mode=RetrievalMode.COMPARE,
                        paper_ids=["paper-b"],
                        dimension="limitations",
                    )
                ],
                coverage_judgments=[
                    CoverageGapJudgment(
                        paper_id="paper-a",
                        dimension="limitations",
                        status="not_stated",
                        reason="The retrieved sections never discuss limitations",
                    )
                ],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="paper-b limitations are now covered",
                covered_requirements=["paper-b limitations"],
                missing_requirements=[],
                retry_recommended=False,
                coverage_judgments=[
                    CoverageGapJudgment(
                        paper_id="paper-b",
                        dimension="limitations",
                        status="covered",
                        reason="The repair query returned the stated failure cases",
                        evidence_ids=[evidence_id("paper-b:chunk:2")],
                    )
                ],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="Two local papers",
                analysis_summary="Limitations were compared where stated.",
                objective_satisfied=True,
                answered_points=["limitations"],
                blocking_gaps=[],
                evidence=[],
                evidence_scope="Two retrieval rounds",
                limitations=["paper-a does not state limitations"],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).side_effect = [
        _report(paper_ids=papers, dimensions=["limitations"], hits=[]),
        _report(
            paper_ids=papers,
            dimensions=["limitations"],
            hits=[_hit("paper-b", "limitations", index=2)],
        ),
    ]
    publisher = CapturingPublisher()

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        _state(
            ReaderTask(
                objective="Compare the limitations of both papers",
                depth=ReaderDepth.DEEP,
                paper_ids=papers,
            )
        ),
        Runtime(context=_context(publisher, ("paper-a", "paper-b"))),
    )

    gap_call = cast(AsyncMock, retriever.retrieve).await_args_list[1]
    assert gap_call.args[0] == "paper-b failure cases and future work"
    # The task scope stays the whole comparison; the sub-question narrows to paper-b.
    assert gap_call.kwargs["paper_ids"] == ["paper-a", "paper-b"]
    assert gap_call.kwargs["queries"][0].paper_ids == ["paper-b"]
    assert gap_call.kwargs["queries"][0].dimension == "limitations"

    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.retrieval_rounds == 2
    assert summary.coverage is not None
    covered = summary.coverage.cell_for("paper-b", "limitations")
    assert covered is not None
    assert covered.status is CoverageStatus.COVERED
    not_stated = summary.coverage.cell_for("paper-a", "limitations")
    assert not_stated is not None
    assert not_stated.status is CoverageStatus.NOT_STATED
    assert not_stated.note is not None
    assert summary.coverage.covered_cell_count == 1
    assert summary.coverage.unresolved_cell_count == 1
    stages = [
        payload["stage"]
        for event_type, payload in publisher.events
        if event_type == "stage.started"
    ]
    assert "reader.retrieve_gaps" in stages


@pytest.mark.asyncio
async def test_reader_forward_declares_a_missing_named_paper() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="method of both papers",
                mode=RetrievalMode.METHOD,
                strategy=RetrievalStrategy.MULTI_PAPER,
                sub_questions=[
                    ReadingSubQuestion(
                        question="method",
                        retrieval_query="method of both papers",
                        mode=RetrievalMode.METHOD,
                        dimension="method",
                    )
                ],
                evidence_requirements=["method per paper"],
                rationale="One query covering both papers",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=False,
                coverage_summary="Only one paper was retrievable",
                covered_requirements=["paper-a method"],
                missing_requirements=["paper-b evidence"],
                retry_recommended=False,
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="Local corpus",
                analysis_summary="Only paper-a had evidence.",
                objective_satisfied=False,
                answered_points=["paper-a method"],
                blocking_gaps=["paper-b is not indexed"],
                evidence=[],
                evidence_scope="One retrieval round",
                limitations=["paper-b evidence is missing"],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    report = _report(
        paper_ids=["paper-a", "paper-b"],
        dimensions=["method"],
        hits=[_hit("paper-a", "method", index=1)],
    ).model_copy(update={"missing_paper_ids": ["paper-b"]})
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).return_value = report
    publisher = CapturingPublisher()

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        _state(
            ReaderTask(
                objective="Compare both papers",
                depth=ReaderDepth.DEEP,
                paper_ids=["paper-a", "paper-b"],
            )
        ),
        Runtime(context=_context(publisher, ("paper-a", "paper-b"))),
    )

    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.missing_paper_ids == ["paper-b"]
    decision_events = [
        payload
        for event_type, payload in publisher.events
        if event_type == "decision.recorded" and payload.get("actor") == "reader"
    ]
    assert decision_events[-1]["missing_paper_ids"] == ["paper-b"]
    assert decision_events[-1]["source"] == DecisionSource.MODEL.value


@pytest.mark.asyncio
async def test_reader_receives_research_plan_dimensions_as_coverage_guidance() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="datasets and metrics",
                mode=RetrievalMode.COMPARE,
                strategy=RetrievalStrategy.MULTI_PAPER,
                coverage_dimensions=["results"],
                evidence_requirements=["results per paper"],
                rationale="The research plan asks for experimental results",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="Results are covered",
                covered_requirements=["results per paper"],
                missing_requirements=[],
                retry_recommended=False,
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="Local corpus",
                analysis_summary="Results were collected.",
                objective_satisfied=True,
                answered_points=["results"],
                blocking_gaps=[],
                evidence=[],
                evidence_scope="One round",
                limitations=[],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).return_value = _report(
        paper_ids=["paper-a", "paper-b"],
        dimensions=["results"],
        hits=[_hit("paper-a", "results", index=1), _hit("paper-b", "results", index=2)],
    )
    publisher = CapturingPublisher()

    await ReaderAgentNode(model, paper_retriever=retriever)(
        _state(
            ReaderTask(
                objective="Compare experimental results",
                depth=ReaderDepth.DEEP,
                paper_ids=["paper-a", "paper-b"],
            ),
            research_plan=ResearchPlan(
                task_type=ResearchTaskType.COMPARISON,
                retrieval_strategy=RetrievalStrategy.MULTI_PAPER,
                answer_dimensions=["results"],
                target_paper_count=2,
                requires_external_search=False,
                requires_local_corpus=True,
                stopping_criteria=["Both papers report results"],
                rationale="A two-paper comparison of reported results",
            ),
        ),
        Runtime(context=_context(publisher, ("paper-a", "paper-b"))),
    )

    plan_input = str(
        cast(AsyncMock, model.generate_structured).await_args_list[0].args[0][-1].content
    )
    assert '"task_type": "comparison"' in plan_input
    assert '"answer_dimensions": ["results"]' in plan_input
    stage_events = [
        payload
        for event_type, payload in publisher.events
        if event_type == "stage.started" and payload.get("actor") == "reader"
    ]
    assert stage_events[0]["research_task_type"] == ResearchTaskType.COMPARISON.value


@pytest.mark.asyncio
async def test_judge_rejects_a_candidate_cell_that_does_not_answer_the_dimension() -> None:
    """A retrieved chunk for a dimension is only a candidate until it is verified."""

    papers = ["paper-a"]
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="reported results of paper-a",
                mode=RetrievalMode.SUMMARY,
                strategy=RetrievalStrategy.SINGLE_PAPER,
                coverage_dimensions=["results"],
                sub_questions=[
                    ReadingSubQuestion(
                        question="results",
                        retrieval_query="reported results of paper-a",
                        mode=RetrievalMode.SUMMARY,
                        dimension="results",
                    )
                ],
                evidence_requirements=["results per paper"],
                rationale="Retrieve the reported results",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=False,
                coverage_summary="The chunk only describes the setup, not any results",
                covered_requirements=[],
                missing_requirements=["reported results"],
                retry_recommended=False,
                coverage_judgments=[
                    CoverageGapJudgment(
                        paper_id="paper-a",
                        dimension="results",
                        status="missing",
                        reason="The retrieved chunk introduces the evaluation setup only",
                    )
                ],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="paper-a",
                analysis_summary="No reported results were found.",
                objective_satisfied=False,
                answered_points=[],
                blocking_gaps=["reported results"],
                evidence=[],
                evidence_scope="One retrieval round",
                limitations=["The paper's results were not retrieved"],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).return_value = _report(
        paper_ids=papers,
        dimensions=["results"],
        hits=[_hit("paper-a", "results", index=1)],
    )
    publisher = CapturingPublisher()

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        _state(
            ReaderTask(
                objective="What results does paper-a report?",
                depth=ReaderDepth.DEEP,
                paper_ids=papers,
            )
        ),
        Runtime(context=_context(publisher, ("paper-a",))),
    )

    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.coverage is not None
    cell = summary.coverage.cell_for("paper-a", "results")
    assert cell is not None
    assert cell.status is CoverageStatus.MISSING
    assert cell.note is not None
    assert summary.coverage.covered_cell_count == 0
    assert summary.coverage.coverage_ratio == 0.0
    # The candidate chunk is still traceable even though the cell was rejected.
    assert cell.evidence_ids == ["paper-a:chunk:1"]
    decision_events = [
        payload
        for event_type, payload in publisher.events
        if event_type == "decision.recorded" and payload.get("actor") == "reader"
    ]
    assert decision_events[-1]["coverage_ratio"] == 0.0
    assert decision_events[-1]["candidate_cell_count"] == 0


@pytest.mark.asyncio
async def test_unjudged_candidate_cells_do_not_count_as_coverage() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="method of paper-a",
                mode=RetrievalMode.METHOD,
                strategy=RetrievalStrategy.SINGLE_PAPER,
                coverage_dimensions=["method"],
                evidence_requirements=["method"],
                rationale="Retrieve the method",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="The method is covered",
                covered_requirements=["method"],
                missing_requirements=[],
                retry_recommended=False,
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="paper-a",
                analysis_summary="The method was described.",
                objective_satisfied=True,
                answered_points=["method"],
                blocking_gaps=[],
                evidence=[],
                evidence_scope="One retrieval round",
                limitations=[],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).return_value = _report(
        paper_ids=["paper-a"],
        dimensions=["method"],
        hits=[_hit("paper-a", "method", index=1)],
    )

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        _state(
            ReaderTask(
                objective="Describe the method",
                depth=ReaderDepth.DEEP,
                paper_ids=["paper-a"],
            )
        ),
        Runtime(context=_context(CapturingPublisher(), ("paper-a",))),
    )

    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.coverage is not None
    cell = summary.coverage.cell_for("paper-a", "method")
    assert cell is not None
    assert cell.status is CoverageStatus.CANDIDATE
    assert summary.coverage.coverage_ratio == 0.0
    assert summary.coverage.candidate_ratio == 1.0


@pytest.mark.asyncio
async def test_covered_verdict_without_valid_evidence_is_downgraded() -> None:
    """A judge cannot mark a cell answered by assertion alone."""

    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="reported results of paper-a",
                mode=RetrievalMode.SUMMARY,
                strategy=RetrievalStrategy.SINGLE_PAPER,
                coverage_dimensions=["results"],
                evidence_requirements=["results"],
                rationale="Retrieve the reported results",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="The judge claims the results are covered",
                covered_requirements=["results"],
                missing_requirements=[],
                retry_recommended=False,
                coverage_judgments=[
                    CoverageGapJudgment(
                        paper_id="paper-a",
                        dimension="results",
                        status="covered",
                        reason="Looks covered by assertion only",
                        evidence_ids=[evidence_id("some-other-paper-chunk")],
                    )
                ],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="paper-a",
                analysis_summary="One chunk about evaluation setup was retrieved.",
                objective_satisfied=False,
                answered_points=[],
                blocking_gaps=["reported results"],
                evidence=[],
                evidence_scope="One retrieval round",
                limitations=["The cited evidence does not belong to this paper"],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).return_value = _report(
        paper_ids=["paper-a"],
        dimensions=["results"],
        hits=[_hit("paper-a", "results", index=1)],
    )

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        _state(
            ReaderTask(
                objective="What results does paper-a report?",
                depth=ReaderDepth.DEEP,
                paper_ids=["paper-a"],
            )
        ),
        Runtime(context=_context(CapturingPublisher(), ("paper-a",))),
    )

    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.coverage is not None
    cell = summary.coverage.cell_for("paper-a", "results")
    assert cell is not None
    assert cell.status is CoverageStatus.MISSING
    assert cell.verified_evidence_ids == []
    assert cell.note is not None
    assert "without evidence" in cell.note
    assert summary.coverage.coverage_ratio == 0.0
    assert summary.coverage.covered_cell_count == 0


@pytest.mark.asyncio
async def test_covered_verdict_with_real_evidence_is_verified() -> None:
    papers = ["paper-a"]
    chunk_id = "paper-a:chunk:1"
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="reported results of paper-a",
                mode=RetrievalMode.SUMMARY,
                strategy=RetrievalStrategy.SINGLE_PAPER,
                coverage_dimensions=["results"],
                evidence_requirements=["results"],
                rationale="Retrieve the reported results",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="The retrieved chunk reports the results table",
                covered_requirements=["results"],
                missing_requirements=[],
                retry_recommended=False,
                coverage_judgments=[
                    CoverageGapJudgment(
                        paper_id="paper-a",
                        dimension="results",
                        status="covered",
                        reason="The chunk reports the measured metrics",
                        evidence_ids=[evidence_id(chunk_id)],
                    )
                ],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="paper-a",
                analysis_summary="The results table was retrieved.",
                objective_satisfied=True,
                answered_points=["results"],
                blocking_gaps=[],
                evidence=[
                    PaperEvidence(
                        evidence_id=evidence_id(chunk_id),
                        claim="The paper reports its results",
                        excerpt="evidence about results",
                    )
                ],
                evidence_scope="One retrieval round",
                limitations=[],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).return_value = _report(
        paper_ids=papers,
        dimensions=["results"],
        hits=[_hit("paper-a", "results", index=1)],
    )

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        _state(
            ReaderTask(
                objective="What results does paper-a report?",
                depth=ReaderDepth.DEEP,
                paper_ids=papers,
            )
        ),
        Runtime(context=_context(CapturingPublisher(), ("paper-a",))),
    )

    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.coverage is not None
    cell = summary.coverage.cell_for("paper-a", "results")
    assert cell is not None
    assert cell.status is CoverageStatus.COVERED
    assert cell.verified_evidence_ids == [evidence_id(chunk_id)]
    assert cell.evidence_ids == [chunk_id]
    assert summary.coverage.coverage_ratio == 1.0
