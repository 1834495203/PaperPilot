from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.papers import Paper, PaperSearchAttempt, PaperSearchResult, PaperSource
from app.domain.ports import EventPublisher, PaperSearchGateway
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
        artifacts=[],
        completed_steps=[],
        decision=None,
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


def test_supervisor_keeps_model_decision_separate_from_policy_override() -> None:
    state = SupervisorState(
        user_request="Is this idea novel?",
        conversation_context="user: Is this idea novel?",
        artifacts=[],
        completed_steps=[],
        decision=None,
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
