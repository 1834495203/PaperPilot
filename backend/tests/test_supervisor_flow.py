from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.application.agent import AgentRunContext
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.enums import EventType
from app.domain.papers import Paper, PaperSearchAttempt, PaperSearchResult, PaperSource
from app.domain.ports import EventPublisher, PaperSearchGateway
from app.domain.rag import (
    RetrievalHit,
    RetrievalMode,
    RetrievalSource,
    TreeRetrievalReport,
)
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.search.tools import AcademicPaperSearchAgentTool
from app.infrastructure.agent.supervisor.analyst_agent import AnalystAgentNode
from app.infrastructure.agent.supervisor.builder import SupervisorGraphBuilder
from app.infrastructure.agent.supervisor.model_gateway import (
    AgentModelGateway,
    ModelUsage,
    StructuredModelResult,
    TextModelResult,
    ToolCallModelResult,
)
from app.infrastructure.agent.supervisor.models import (
    AgentName,
    AnalystTask,
    DecisionAssessment,
    DecisionSource,
    PaperAssessment,
    PaperRelevance,
    ReaderDepth,
    ReaderTask,
    ReadingEvidenceAssessment,
    ReadingReport,
    SearchScreening,
    SearchTask,
    SupervisorDecision,
    WriterTask,
)
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.search_agent import SearchAgentNode
from app.infrastructure.agent.supervisor.state import SupervisorState
from app.infrastructure.agent.supervisor.supervisor_node import SupervisorNode
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode


def _unused_search_node(model: AgentModelGateway) -> SearchAgentNode:
    gateway = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    return SearchAgentNode(
        model=model,
        tool=AcademicPaperSearchAgentTool(gateway),
        recorder=cast(
            AgentExecutionRecorder,
            create_autospec(AgentExecutionRecorder, instance=True),
        ),
    )


class CapturingPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, JsonValue]]] = []

    async def publish(
        self,
        event_type: str,
        payload: dict[str, JsonValue],
    ) -> None:
        self.events.append((event_type, payload))


@pytest.mark.asyncio
async def test_supervisor_routes_search_result_back_to_writer() -> None:
    model = cast(
        AgentModelGateway,
        create_autospec(AgentModelGateway, instance=True),
    )
    structured = cast(AsyncMock, model.generate_structured)
    structured.side_effect = [
        StructuredModelResult(
            value=SupervisorDecision(
                assessment=DecisionAssessment(
                    observations=["The user asks for an external paper"],
                    missing_information=["No paper evidence is available yet"],
                    decision_summary="External paper evidence is required",
                ),
                task=SearchTask(
                    objective="Find a relevant paper",
                    query="retrieval augmented generation hallucination evaluation",
                ),
            ),
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        StructuredModelResult(
            value=SearchScreening(
                screening_summary="The returned paper directly matches the topic",
                assessments=[
                    PaperAssessment(
                        paper_id="arxiv:2401.00001",
                        relevance=PaperRelevance.DIRECT,
                        relevance_reason="It evaluates hallucinations in RAG",
                        matched_topics=["RAG", "hallucination evaluation"],
                    )
                ],
                continue_search=False,
            ),
            usage=ModelUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        ),
        StructuredModelResult(
            value=SupervisorDecision(
                assessment=DecisionAssessment(
                    observations=["The search returned a relevant paper"],
                    missing_information=[],
                    decision_summary="The search result satisfies the request",
                ),
                task=WriterTask(objective="Present the discovered paper"),
            ),
            usage=ModelUsage(input_tokens=12, output_tokens=4, total_tokens=16),
        ),
    ]
    tool_call = cast(AsyncMock, model.generate_tool_call)
    tool_call.return_value = ToolCallModelResult(
        call_id="call-1",
        tool_name="search_academic_papers",
        arguments={
            "query": "retrieval augmented generation hallucination evaluation",
            "max_results": 3,
            "sort_by": "relevance",
            "decision_summary": "The focused query matches the requested RAG evaluation topic",
        },
        usage=ModelUsage(input_tokens=8, output_tokens=4, total_tokens=12),
    )
    text = cast(AsyncMock, model.generate_text)
    final_message = AIMessage(
        content="Found one relevant paper.",
        usage_metadata={"input_tokens": 9, "output_tokens": 6, "total_tokens": 15},
    )
    text.return_value = TextModelResult(
        text="Found one relevant paper.",
        usage=ModelUsage(input_tokens=9, output_tokens=6, total_tokens=15),
        message=final_message,
    )

    paper_search = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    paper = Paper.model_validate(
        {
            "paper_id": "arxiv:2401.00001",
            "source": "arxiv",
            "arxiv_id": "2401.00001",
            "external_ids": {"arxiv": "2401.00001"},
            "title": "Evaluating Hallucinations in RAG",
            "summary": "A grounded evaluation study.",
            "authors": ["Ada Example"],
            "published_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "landing_page_url": "https://arxiv.org/abs/2401.00001",
            "pdf_url": "https://arxiv.org/pdf/2401.00001",
        }
    )
    cast(AsyncMock, paper_search.search).return_value = PaperSearchResult(
        papers=[paper],
        provider=PaperSource.ARXIV,
        attempts=[
            PaperSearchAttempt(
                provider=PaperSource.ARXIV,
                status="completed",
                result_count=1,
            )
        ],
    )
    recorder = cast(
        AgentExecutionRecorder,
        create_autospec(AgentExecutionRecorder, instance=True),
    )
    cast(AsyncMock, recorder.record_tool_execution).return_value = ToolMessage(
        content="[]",
        tool_call_id="call-1",
        name="search_academic_papers",
    )
    cast(AsyncMock, recorder.record_assistant_message).return_value = uuid4()
    graph = SupervisorGraphBuilder(
        supervisor=SupervisorNode(model, max_steps=6),
        search=SearchAgentNode(
            model=model,
            tool=AcademicPaperSearchAgentTool(paper_search),
            recorder=recorder,
        ),
        reader=ReaderAgentNode(model),
        analyst=AnalystAgentNode(model),
        writer=WriterAgentNode(model=model, recorder=recorder),
    ).build()
    publisher = CapturingPublisher()
    initial_state = SupervisorState(
        user_request="Find a paper about RAG hallucination evaluation",
        conversation_context="user: Find a paper about RAG hallucination evaluation",
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

    result = await graph.ainvoke(
        initial_state,
        context=AgentRunContext(
            conversation_id=uuid4(),
            run_id=uuid4(),
            publisher=publisher,
        ),
    )

    assert [step.agent for step in result["completed_steps"]] == [
        AgentName.SEARCH,
        AgentName.WRITER,
    ]
    assert result["llm_calls"] == 5
    assert result["tool_calls"] == 1
    assert result["total_tokens"] == 63
    cast(AsyncMock, recorder.record_tool_execution).assert_awaited_once()
    cast(AsyncMock, recorder.record_assistant_message).assert_awaited_once()
    stage_names = [
        payload["stage"]
        for event_type, payload in publisher.events
        if event_type == EventType.STAGE_STARTED.value
    ]
    assert stage_names == ["supervisor", "search", "supervisor", "writer"]
    supervisor_decisions = [
        payload
        for event_type, payload in publisher.events
        if event_type == EventType.DECISION_RECORDED.value and payload.get("actor") == "supervisor"
    ]
    assert supervisor_decisions[0]["source"] == DecisionSource.MODEL.value
    assert supervisor_decisions[0]["next_agent"] == AgentName.SEARCH.value
    assert supervisor_decisions[0]["observations"] == ["The user asks for an external paper"]


@pytest.mark.asyncio
async def test_quick_reader_flow_reaches_writer_without_another_supervisor_call() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    structured = cast(AsyncMock, model.generate_structured)
    structured.side_effect = [
        StructuredModelResult(
            value=SupervisorDecision(
                assessment=DecisionAssessment(
                    observations=["The local corpus can answer this narrow question"],
                    missing_information=["How parsing failures are mitigated"],
                    decision_summary="Read the local corpus once",
                ),
                task=ReaderTask(
                    objective="How does the system mitigate PDF parsing failures?",
                    depth=ReaderDepth.QUICK,
                ),
            ),
            usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        ),
        StructuredModelResult(
            value=ReadingEvidenceAssessment(
                evidence_sufficient=True,
                coverage_summary="The retrieved chunk answers the question",
                covered_requirements=["Parsing fallback"],
                missing_requirements=[],
                retry_recommended=False,
            ),
            usage=ModelUsage(input_tokens=8, output_tokens=4, total_tokens=12),
        ),
        StructuredModelResult(
            value=ReadingReport(
                paper_or_material="Indexed corpus",
                analysis_summary="Parsing failures fall back to layout-aware extraction.",
                objective_satisfied=True,
                answered_points=["Parsing fallback"],
                blocking_gaps=[],
                evidence=[],
                evidence_scope="One retrieved chunk",
                limitations=[],
            ),
            usage=ModelUsage(input_tokens=6, output_tokens=3, total_tokens=9),
        ),
    ]
    text = cast(AsyncMock, model.generate_text)
    text.return_value = TextModelResult(
        text="The system falls back to layout-aware extraction.",
        usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
        message=AIMessage(content="The system falls back to layout-aware extraction."),
    )
    retriever = cast(TreeRagRetriever, create_autospec(TreeRagRetriever, instance=True))
    cast(AsyncMock, retriever.retrieve).return_value = TreeRetrievalReport(
        query="How does the system mitigate PDF parsing failures?",
        mode=RetrievalMode.METHOD,
        paper_ids=[],
        searched_globally=True,
        candidate_paper_ids=["paperqa"],
        initial_hit_count=1,
        expanded_candidate_count=0,
        hits=[
            RetrievalHit(
                rank=1,
                node_id="paperqa:chunk:1",
                paper_id="paperqa",
                section_path=["Methods"],
                page_start=3,
                page_end=3,
                text="Parsing fallback evidence",
                vector_score=0.9,
                ranking_score=0.9,
                source=RetrievalSource.VECTOR,
            )
        ],
    )
    recorder = cast(
        AgentExecutionRecorder, create_autospec(AgentExecutionRecorder, instance=True)
    )
    cast(AsyncMock, recorder.record_assistant_message).return_value = uuid4()
    graph = SupervisorGraphBuilder(
        supervisor=SupervisorNode(model, max_steps=6),
        search=_unused_search_node(model),
        reader=ReaderAgentNode(model, paper_retriever=retriever),
        analyst=AnalystAgentNode(model),
        writer=WriterAgentNode(model=model, recorder=recorder),
    ).build()
    publisher = CapturingPublisher()
    initial_state = SupervisorState(
        user_request="How does the system mitigate PDF parsing failures?",
        conversation_context="user: How does the system mitigate PDF parsing failures?",
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

    result = await graph.ainvoke(
        initial_state,
        context=AgentRunContext(
            conversation_id=uuid4(),
            run_id=uuid4(),
            publisher=publisher,
            local_corpus_available=True,
        ),
    )

    assert [step.agent for step in result["completed_steps"]] == [
        AgentName.READER,
        AgentName.WRITER,
    ]
    # One supervisor decision, two reader calls, one writer call: the quick exit
    # routes to Writer as policy instead of paying for another supervisor model call.
    assert structured.await_count == 3
    assert text.await_count == 1
    stage_names = [
        payload["stage"]
        for event_type, payload in publisher.events
        if event_type == EventType.STAGE_STARTED.value
    ]
    assert stage_names == ["supervisor", "reader", "reader.assess", "writer"]
    writer_prompt = str(text.await_args.args[0][-1].content)
    assert "Answer the user's exact question directly and concisely" in writer_prompt


def test_supervisor_keeps_model_decision_separate_from_policy_override() -> None:
    state = SupervisorState(
        user_request="Is this idea novel?",
        conversation_context="user: Is this idea novel?",
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
    model_decision = SupervisorDecision(
        assessment=DecisionAssessment(
            observations=["The user requests novelty analysis"],
            missing_information=["No prior-art evidence is available"],
            decision_summary="Novelty analysis is needed",
        ),
        task=AnalystTask(
            objective="Assess novelty",
            source_artifact_ids=[uuid4()],
        ),
    )

    resolution = SupervisorNode._normalize_decision(state, model_decision)

    assert model_decision.task.agent is AgentName.ANALYST
    assert resolution.decision.task.agent is AgentName.SEARCH
    assert resolution.adjustments[-1].rule == "analyst_requires_sources"
