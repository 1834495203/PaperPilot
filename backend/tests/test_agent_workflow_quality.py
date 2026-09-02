import json
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, create_autospec, patch
from uuid import uuid4

import pytest
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.application.paper_library import PaperLibraryService
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.papers import Paper
from app.domain.ports import EventPublisher, PaperSearchGateway
from app.domain.rag import (
    IndexedPaper,
    PaperMetadata,
    RetrievalHit,
    RetrievalMode,
    RetrievalSource,
    TreeRetrievalReport,
)
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.search.tools import ArxivSearchAgentTool
from app.infrastructure.agent.supervisor.analyst_agent import AnalystAgentNode
from app.infrastructure.agent.supervisor.model_gateway import (
    AgentModelGateway,
    ModelUsage,
    StructuredModelResult,
    ToolCallModelResult,
)
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    AgentName,
    AnalysisReport,
    AnalystAgentSummary,
    AnalystTask,
    DecisionAssessment,
    PaperAssessment,
    PaperRelevance,
    ReaderAgentSummary,
    ReaderTask,
    ReadingEvidenceAssessment,
    ReadingPlan,
    ReadingReport,
    SearchAgentSummary,
    SearchReport,
    SearchScreening,
    SearchStatus,
    SearchTask,
    SupervisorDecision,
    WriterTask,
)
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.search_agent import SearchAgentNode
from app.infrastructure.agent.supervisor.state import SupervisorState
from app.infrastructure.agent.supervisor.supervisor_node import SupervisorNode
from app.infrastructure.agent.supervisor.support import (
    render_artifacts,
    render_supervisor_context,
)
from app.infrastructure.tools.arxiv import ArxivApiError
from app.infrastructure.tools.pdf import ArxivPdfDocumentGateway, PdfDocumentError


class CapturingPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, JsonValue]]] = []

    async def publish(self, event_type: str, payload: dict[str, JsonValue]) -> None:
        self.events.append((event_type, payload))


def make_paper(arxiv_id: str = "2401.00001") -> Paper:
    return Paper.model_validate(
        {
            "arxiv_id": arxiv_id,
            "title": "Directly Relevant Paper",
            "summary": "A paper about efficient LLM agent collaboration.",
            "authors": ["Ada Example"],
            "published_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        }
    )


def empty_state(*, artifacts: list[AgentArtifact] | None = None) -> SupervisorState:
    return SupervisorState(
        user_request="Find five directly relevant papers",
        conversation_context="user: Find five directly relevant papers",
        artifacts=artifacts or [],
        completed_steps=[],
        decision=None,
        step_count=0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        llm_calls=0,
        tool_calls=0,
    )


def test_supervisor_context_contains_only_agent_summaries() -> None:
    old = AgentArtifact(
        title="Old search",
        supervisor_summary=SearchAgentSummary(
            summary="Old summary",
            attempted_queries=["old query"],
            direct_count=1,
            adjacent_count=0,
            irrelevant_count=0,
            papers=[],
        ),
        content="x" * 30_000,
    )
    latest = AgentArtifact(
        title="Latest analysis",
        supervisor_summary=AnalystAgentSummary(
            summary="Only two papers are directly relevant",
            finding_count=2,
            research_gap_count=1,
            novelty_assessment="Incomplete evidence",
            research_gaps=["Three papers are missing"],
            unresolved_questions=[],
        ),
        content="LATEST ANALYSIS CONTENT",
    )

    rendered = json.loads(render_supervisor_context([old, latest]))

    assert len(rendered) == 2
    assert rendered[-1]["id"] == str(latest.id)
    assert rendered[-1]["summary"]["summary"] == "Only two papers are directly relevant"
    assert "LATEST ANALYSIS CONTENT" not in render_supervisor_context([old, latest])
    assert "x" * 100 not in render_supervisor_context([old, latest])


def test_downstream_artifact_transfer_does_not_truncate_content() -> None:
    marker = "FULL_REPORT_END"
    artifact = AgentArtifact(
        title="Long search report",
        supervisor_summary=SearchAgentSummary(
            summary="Search completed",
            attempted_queries=["query"],
            direct_count=1,
            adjacent_count=0,
            irrelevant_count=0,
            papers=[],
        ),
        content=("x" * 100_000) + marker,
    )

    rendered = render_artifacts([artifact])

    assert marker in rendered
    assert len(json.loads(rendered)[0]["content"]) == 100_000 + len(marker)


@pytest.mark.asyncio
async def test_analyst_receives_complete_reader_report_but_not_global_request() -> None:
    marker = "READER_REPORT_END"
    reading = AgentArtifact(
        title="Reading report",
        supervisor_summary=ReaderAgentSummary(
            summary="The assigned paper was read",
            paper_or_material="Paper A",
            objective_satisfied=True,
            answered_points=["Agent collaboration method"],
            blocking_gaps=[],
            evidence_scope="Full PDF",
            limitations=[],
            pdf_truncated=False,
        ),
        content=("detailed-reader-report-" * 6_000) + marker,
    )
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).return_value = StructuredModelResult(
        value=AnalysisReport(
            analysis_type="comparison",
            analysis_summary="Compared the supplied report",
            comparison_dimensions=["method"],
            findings=[],
            research_gaps=[],
            novelty_assessment="Not assessed",
            unresolved_questions=[],
        ),
        usage=ModelUsage(total_tokens=2),
    )
    state = empty_state(artifacts=[reading])
    state["user_request"] = "GLOBAL REQUEST SHOULD STAY WITH SUPERVISOR"
    state["decision"] = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=["A reading report is ready"],
            missing_information=["Cross-paper synthesis"],
            decision_summary="Analyze the selected report",
        ),
        task=AnalystTask(
            objective="Analyze the supplied reading report",
            source_artifact_ids=[reading.id],
        ),
    )

    await AnalystAgentNode(model)(
        state,
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
            )
        ),
    )

    messages = cast(AsyncMock, model.generate_structured).await_args.args[0]
    analyst_input = str(messages[-1].content)
    assert marker in analyst_input
    assert "GLOBAL REQUEST SHOULD STAY WITH SUPERVISOR" not in analyst_input


def test_reader_task_rejects_fields_from_other_agent_tasks() -> None:
    with pytest.raises(ValueError, match="source_artifact_ids"):
        ReaderTask.model_validate(
            {
                "agent": "reader",
                "objective": "Read one paper",
                "source_artifact_id": str(uuid4()),
                "paper_id": "2401.00001",
                "source_artifact_ids": [],
            }
        )


def test_supervisor_respects_writer_decision_when_evidence_is_incomplete() -> None:
    decision = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=["Only two papers are directly relevant"],
            missing_information=["Three more directly relevant papers are required"],
            decision_summary="The current evidence is incomplete",
        ),
        task=WriterTask(objective="Answer with five papers"),
    )

    resolution = SupervisorNode._normalize_decision(empty_state(), decision)

    assert resolution.decision.task.agent is AgentName.WRITER
    assert resolution.adjustments == []


def test_search_screening_filters_unknown_ids_and_marks_omissions() -> None:
    papers = {"2401.00001": make_paper(), "2401.00002": make_paper("2401.00002")}
    screening = SearchScreening(
        screening_summary="One candidate is directly relevant",
        assessments=[
            PaperAssessment(
                arxiv_id="2401.00001",
                relevance=PaperRelevance.DIRECT,
                relevance_reason="It studies LLM agent collaboration",
                matched_topics=["LLM agents", "collaboration"],
            ),
            PaperAssessment(
                arxiv_id="unknown",
                relevance=PaperRelevance.DIRECT,
                relevance_reason="Not returned by the tool",
                matched_topics=[],
            ),
        ],
        continue_search=False,
    )

    normalized = SearchAgentNode._normalize_screening(screening, papers)
    by_id = {item.arxiv_id: item for item in normalized.assessments}

    assert set(by_id) == set(papers)
    assert by_id["2401.00001"].relevance is PaperRelevance.DIRECT
    assert by_id["2401.00002"].relevance is PaperRelevance.IRRELEVANT


@pytest.mark.asyncio
async def test_search_agent_executes_rewritten_query_after_weak_screening() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_tool_call).side_effect = [
        ToolCallModelResult(
            call_id="call-1",
            tool_name="search_arxiv",
            arguments={
                "query": "multi-agent collaboration",
                "max_results": 10,
                "sort_by": "relevance",
                "decision_summary": "Start with broad collaboration terms",
            },
            usage=ModelUsage(total_tokens=5),
        ),
        ToolCallModelResult(
            call_id="call-2",
            tool_name="search_arxiv",
            arguments={
                "query": "large language model agent communication protocol",
                "max_results": 10,
                "sort_by": "relevance",
                "decision_summary": "Exclude traditional MARL by adding LLM-specific concepts",
            },
            usage=ModelUsage(total_tokens=5),
        ),
    ]
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=SearchScreening(
                screening_summary="The first result is traditional MARL, not LLM agents",
                assessments=[
                    PaperAssessment(
                        arxiv_id="2401.00001",
                        relevance=PaperRelevance.IRRELEVANT,
                        relevance_reason="It does not study language-model agents",
                        matched_topics=["multi-agent"],
                    )
                ],
                continue_search=True,
                rewritten_query="large language model agent communication protocol",
            ),
            usage=ModelUsage(total_tokens=3),
        ),
        StructuredModelResult(
            value=SearchScreening(
                screening_summary="The rewritten query returned a direct LLM-agent paper",
                assessments=[
                    PaperAssessment(
                        arxiv_id="2401.00001",
                        relevance=PaperRelevance.IRRELEVANT,
                        relevance_reason="It does not study language-model agents",
                        matched_topics=["multi-agent"],
                    ),
                    PaperAssessment(
                        arxiv_id="2401.00002",
                        relevance=PaperRelevance.DIRECT,
                        relevance_reason="It studies communication protocols for LLM agents",
                        matched_topics=["LLM agents", "communication protocol"],
                    ),
                ],
                continue_search=False,
            ),
            usage=ModelUsage(total_tokens=3),
        ),
    ]
    paper_search = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    cast(AsyncMock, paper_search.search).side_effect = [
        [make_paper("2401.00001")],
        [make_paper("2401.00002")],
    ]
    recorder = cast(
        AgentExecutionRecorder,
        create_autospec(AgentExecutionRecorder, instance=True),
    )
    node = SearchAgentNode(
        model=model,
        tool=ArxivSearchAgentTool(paper_search),
        recorder=recorder,
        max_iterations=2,
    )
    state = empty_state()
    state["decision"] = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=[],
            missing_information=["Direct LLM-agent evidence"],
            decision_summary="Search is required",
        ),
        task=SearchTask(
            objective="Find papers about efficient LLM-agent collaboration",
            query="multi-agent collaboration",
        ),
    )
    publisher = CapturingPublisher()

    update = await node(
        state,
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=publisher,
            )
        ),
    )

    assert update["tool_calls"] == 2
    assert update["llm_calls"] == 4
    search_report = SearchReport.model_validate_json(update["artifacts"][-1].content)
    assert search_report.attempted_queries == [
        "multi-agent collaboration",
        "large language model agent communication protocol",
    ]
    assert any(
        item.arxiv_id == "2401.00002" and item.relevance is PaperRelevance.DIRECT
        for item in search_report.assessments
    )
    assert cast(AsyncMock, paper_search.search).await_count == 2


@pytest.mark.asyncio
async def test_search_failure_is_visible_to_supervisor_and_stops_internal_loop() -> None:
    model = cast(
        AgentModelGateway,
        create_autospec(AgentModelGateway, instance=True),
    )
    cast(AsyncMock, model.generate_tool_call).return_value = ToolCallModelResult(
        call_id="rate-limited-call",
        tool_name="search_arxiv",
        arguments={
            "query": "LLM agent collaboration",
            "max_results": 10,
            "sort_by": "relevance",
            "decision_summary": "Find direct literature",
        },
        usage=ModelUsage(total_tokens=2),
    )
    paper_search = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    cast(AsyncMock, paper_search.search).side_effect = ArxivApiError(
        "arXiv rate limit reached (HTTP 429)",
        category="rate_limited",
        retryable=True,
        status_code=429,
        retry_after_seconds=15,
    )
    recorder = cast(
        AgentExecutionRecorder,
        create_autospec(AgentExecutionRecorder, instance=True),
    )
    node = SearchAgentNode(
        model=model,
        tool=ArxivSearchAgentTool(paper_search),
        recorder=recorder,
        max_iterations=2,
    )
    state = empty_state()
    state["decision"] = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=[],
            missing_information=["External evidence"],
            decision_summary="Search externally",
        ),
        task=SearchTask(
            objective="Find papers about LLM-agent collaboration",
            query="LLM agent collaboration",
        ),
    )

    update = await node(
        state,
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
            )
        ),
    )

    artifact = update["artifacts"][-1]
    assert artifact.supervisor_summary.status is SearchStatus.RATE_LIMITED
    assert artifact.supervisor_summary.failures[0].status_code == 429
    assert "HTTP 429" in artifact.supervisor_summary.summary
    assert cast(AsyncMock, paper_search.search).await_count == 1
    assert cast(AsyncMock, model.generate_tool_call).await_count == 1
    cast(AsyncMock, model.generate_structured).assert_not_awaited()


def test_reader_selects_requested_pdf_from_search_report() -> None:
    paper = make_paper()
    report = SearchReport(
        attempted_queries=["LLM agent collaboration"],
        papers=[paper],
        assessments=[
            PaperAssessment(
                arxiv_id=paper.arxiv_id,
                relevance=PaperRelevance.DIRECT,
                relevance_reason="Direct match",
                matched_topics=["collaboration"],
            )
        ],
        screening_summary="One direct match",
    )
    artifact = AgentArtifact(
        title="Search",
        supervisor_summary=SearchAgentSummary(
            summary="One direct match",
            attempted_queries=["LLM agent collaboration"],
            direct_count=1,
            adjacent_count=0,
            irrelevant_count=0,
            papers=[],
        ),
        content=report.model_dump_json(),
    )

    selected = ReaderAgentNode._select_pdf_candidate([artifact], paper.arxiv_id)

    assert selected is not None
    assert selected.paper.arxiv_id == paper.arxiv_id


@pytest.mark.asyncio
async def test_reader_receives_only_its_single_paper_scope() -> None:
    requested = make_paper("2401.00001")
    unrelated = make_paper("2401.00002")
    report = SearchReport(
        attempted_queries=["LLM agent collaboration"],
        papers=[requested, unrelated],
        assessments=[],
        screening_summary="Two candidates",
    )
    source = AgentArtifact(
        title="Search",
        supervisor_summary=SearchAgentSummary(
            summary="Two candidates",
            attempted_queries=["LLM agent collaboration"],
            direct_count=2,
            adjacent_count=0,
            irrelevant_count=0,
            papers=[],
        ),
        content=report.model_dump_json(),
    )
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=False,
                evidence_requirements=["Method summary from supplied metadata"],
                rationale="The supplied abstract is sufficient for the focused objective",
            ),
            usage=ModelUsage(total_tokens=1),
        ),
        StructuredModelResult(
            value=ReadingReport(
            paper_or_material=requested.title,
            analysis_summary="Read the assigned paper",
            answer_material="The paper uses a structured communication protocol.",
            objective_satisfied=True,
            answered_points=["Method"],
            blocking_gaps=[],
            evidence=[],
            evidence_scope="Metadata only",
            limitations=["One benchmark"],
            ),
            usage=ModelUsage(total_tokens=3),
        ),
    ]
    state = empty_state(artifacts=[source])
    state["user_request"] = "GLOBAL REQUEST: compare five papers"
    state["decision"] = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=["One paper needs reading"],
            missing_information=["Its method"],
            decision_summary="Read one paper",
        ),
        task=ReaderTask(
            objective="Summarize this paper's method",
            source_artifact_id=source.id,
            paper_id=requested.arxiv_id,
        ),
    )

    await ReaderAgentNode(model)(
        state,
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
            )
        ),
    )

    messages = cast(AsyncMock, model.generate_structured).await_args.args[0]
    reader_input = str(messages[-1].content)
    assert "GLOBAL REQUEST" not in reader_input
    assert requested.arxiv_id in reader_input
    assert unrelated.arxiv_id not in reader_input


@pytest.mark.asyncio
async def test_reader_retrieves_selected_local_paper_as_structured_evidence() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="How is the tree index built?",
                mode=RetrievalMode.METHOD,
                evidence_requirements=["Tree construction method"],
                rationale="The objective asks about index construction",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="The method evidence is covered",
                covered_requirements=["Tree construction method"],
                missing_requirements=[],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="Local TreeRAG Paper",
                analysis_summary="The retrieved method evidence was summarized",
                answer_material="The paper builds a hierarchy-aware tree index.",
                objective_satisfied=True,
                answered_points=["Tree construction method"],
                blocking_gaps=[],
                evidence=[],
                evidence_scope="Retrieved chunks from page 4",
                limitations=[],
            ),
            usage=ModelUsage(total_tokens=3),
        ),
    ]
    retriever = cast(
        TreeRagRetriever,
        create_autospec(TreeRagRetriever, instance=True),
    )
    cast(AsyncMock, retriever.retrieve).return_value = TreeRetrievalReport(
        query="How is the tree index built?",
        mode=RetrievalMode.METHOD,
        paper_ids=["local-paper"],
        initial_hit_count=4,
        expanded_candidate_count=2,
        hits=[
            RetrievalHit(
                rank=1,
                node_id="local-paper:chunk:1",
                paper_id="local-paper",
                section_path=["3 Method"],
                page_start=4,
                page_end=4,
                text="The index preserves ancestor titles as embedding prefixes.",
                vector_score=0.9,
                ranking_score=0.92,
                source=RetrievalSource.VECTOR,
            )
        ],
    )
    state = empty_state()
    state["decision"] = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=["A local paper is selected"],
            missing_information=["Its indexing method"],
            decision_summary="Retrieve the selected paper",
        ),
        task=ReaderTask(
            objective="Summarize this paper's indexing method",
            paper_id="local-paper",
        ),
    )

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        state,
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
                paper_ids=("local-paper",),
            )
        ),
    )

    reader_input = str(
        cast(AsyncMock, model.generate_structured).await_args.args[0][-1].content
    )
    assert "ancestor titles as embedding prefixes" in reader_input
    assert '"page_start": 4' in reader_input
    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.retrieval_hit_count == 1
    assert summary.retrieval_rounds == 1
    assert summary.attempted_queries == ["How is the tree index built?"]
    assert update["tool_calls"] == 1
    assert update["llm_calls"] == 3


@pytest.mark.asyncio
async def test_reader_subgraph_rewrites_query_when_evidence_is_incomplete() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    reading_report = ReadingReport(
        paper_or_material="Local Paper",
        analysis_summary="Combined method and experiment evidence",
        answer_material="Tree retrieval is evaluated on a benchmark and improves recall.",
        objective_satisfied=True,
        answered_points=["method", "experiments"],
        blocking_gaps=[],
        evidence=[],
        evidence_scope="Two retrieval rounds",
        limitations=[],
    )
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=True,
                query="tree retrieval method",
                mode=RetrievalMode.METHOD,
                evidence_requirements=["method", "experiments"],
                rationale="Both method and evaluation are required",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=False,
                coverage_summary="Method is covered but experiments are missing",
                covered_requirements=["method"],
                missing_requirements=["experiments"],
                next_query="experimental setup datasets metrics results",
                next_mode=RetrievalMode.SUMMARY,
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="Method and experiments are now covered",
                covered_requirements=["method", "experiments"],
                missing_requirements=[],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=reading_report,
            usage=ModelUsage(total_tokens=3),
        ),
    ]
    retriever = cast(
        TreeRagRetriever,
        create_autospec(TreeRagRetriever, instance=True),
    )
    cast(AsyncMock, retriever.retrieve).side_effect = [
        TreeRetrievalReport(
            query="tree retrieval method",
            mode=RetrievalMode.METHOD,
            paper_ids=["local-paper"],
            initial_hit_count=1,
            expanded_candidate_count=0,
            hits=[
                RetrievalHit(
                    rank=1,
                    node_id="method-chunk",
                    paper_id="local-paper",
                    section_path=["3 Method"],
                    page_start=3,
                    page_end=3,
                    text="Method evidence",
                    vector_score=0.9,
                    ranking_score=0.9,
                    source=RetrievalSource.VECTOR,
                )
            ],
        ),
        TreeRetrievalReport(
            query="experimental setup datasets metrics results",
            mode=RetrievalMode.SUMMARY,
            paper_ids=["local-paper"],
            initial_hit_count=1,
            expanded_candidate_count=0,
            hits=[
                RetrievalHit(
                    rank=1,
                    node_id="experiment-chunk",
                    paper_id="local-paper",
                    section_path=["4 Experiments"],
                    page_start=6,
                    page_end=6,
                    text="Experiment evidence",
                    vector_score=0.85,
                    ranking_score=0.86,
                    source=RetrievalSource.VECTOR,
                )
            ],
        ),
    ]
    state = empty_state()
    state["decision"] = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=["One local paper is selected"],
            missing_information=["Method and experiments"],
            decision_summary="Read the selected paper",
        ),
        task=ReaderTask(
            objective="Summarize the method and experimental evaluation",
            paper_id="local-paper",
        ),
    )

    update = await ReaderAgentNode(model, paper_retriever=retriever)(
        state,
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
                paper_ids=("local-paper",),
            )
        ),
    )

    retrieve_calls = cast(AsyncMock, retriever.retrieve).await_args_list
    assert [call.args[0] for call in retrieve_calls] == [
        "tree retrieval method",
        "experimental setup datasets metrics results",
    ]
    summary = update["artifacts"][-1].supervisor_summary
    assert isinstance(summary, ReaderAgentSummary)
    assert summary.retrieval_rounds == 2
    assert summary.retrieval_hit_count == 2
    assert update["tool_calls"] == 2
    assert update["llm_calls"] == 4


@pytest.mark.asyncio
async def test_reader_skips_rag_when_metadata_already_answers_the_objective() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(
            value=ReadingPlan(
                needs_retrieval=False,
                evidence_requirements=["Paper authors"],
                rationale="Author names are already present in authoritative metadata",
            ),
            usage=ModelUsage(total_tokens=2),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="Local Paper",
                analysis_summary="The author question is answered by metadata",
                answer_material="The authors are Ada Lovelace and Alan Turing.",
                objective_satisfied=True,
                answered_points=["Authors"],
                blocking_gaps=[],
                evidence=[],
                evidence_scope="Local paper metadata",
                limitations=[],
            ),
            usage=ModelUsage(total_tokens=2),
        ),
    ]
    retriever = cast(
        TreeRagRetriever,
        create_autospec(TreeRagRetriever, instance=True),
    )
    library = cast(
        PaperLibraryService,
        create_autospec(PaperLibraryService, instance=True),
    )
    cast(AsyncMock, library.get_paper).return_value = IndexedPaper(
        paper_id="local-paper",
        metadata=PaperMetadata(
            title="Local Paper",
            authors=["Ada Lovelace", "Alan Turing"],
        ),
        original_filename="paper.pdf",
        content_sha256="a" * 64,
        page_count=4,
        section_count=3,
        node_count=8,
        chunk_count=4,
    )
    state = empty_state()
    state["decision"] = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=["A local paper is selected"],
            missing_information=["Authors"],
            decision_summary="Read the paper metadata",
        ),
        task=ReaderTask(objective="Who are the authors?", paper_id="local-paper"),
    )

    update = await ReaderAgentNode(
        model,
        paper_retriever=retriever,
        paper_library=library,
    )(
        state,
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
                paper_ids=("local-paper",),
            )
        ),
    )

    cast(AsyncMock, retriever.retrieve).assert_not_awaited()
    assert update["tool_calls"] == 0
    assert update["llm_calls"] == 2


def test_supervisor_allows_reader_for_a_selected_local_paper() -> None:
    decision = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=["A local paper is selected"],
            missing_information=["Its method"],
            decision_summary="Read the local paper",
        ),
        task=ReaderTask(
            objective="Read one local paper",
            paper_id="local-paper",
        ),
    )

    resolution = SupervisorNode._normalize_decision(
        empty_state(),
        decision,
        local_paper_ids={"local-paper"},
    )

    assert isinstance(resolution.decision.task, ReaderTask)
    assert resolution.decision.task.paper_id == "local-paper"


def test_pdf_gateway_rejects_non_arxiv_urls() -> None:
    with pytest.raises(PdfDocumentError, match="Only arxiv.org"):
        ArxivPdfDocumentGateway._validated_url("https://example.com/paper.pdf")


def test_pdf_gateway_extracts_page_marked_text() -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        is_encrypted = False
        pages = [FakePage("Introduction"), FakePage("Method and experiments")]

    gateway = ArxivPdfDocumentGateway(
        timeout_seconds=10,
        max_bytes=1_000_000,
        max_pages=10,
        max_characters=10_000,
    )

    with patch(
        "app.infrastructure.tools.pdf.PdfReader",
        return_value=FakeReader(),
    ):
        document = gateway._extract(b"%PDF-fake", "https://arxiv.org/pdf/2401.00001")

    assert document.page_count == 2
    assert document.extracted_pages == 2
    assert "--- PAGE 1 ---" in document.text
    assert "--- PAGE 2 ---" in document.text
    assert document.truncated is False
