from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.domain.ports import (
    ConversationStore,
    EventPublisher,
    PaperDocumentGateway,
    PaperSearchGateway,
)
from app.domain.rag import RetrievalStrategy
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
)
from app.infrastructure.agent.supervisor.models import (
    DecisionAssessment,
    DecisionSource,
    ReaderDepth,
    ReaderTask,
    ResearchPlan,
    ResearchTaskType,
    SearchTask,
    SupervisorDecision,
    WriterTask,
)
from app.infrastructure.agent.supervisor.planner_agent import ResearchPlannerNode
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.search_agent import SearchAgentNode
from app.infrastructure.agent.supervisor.state import SupervisorState
from app.infrastructure.agent.supervisor.supervisor_node import SupervisorNode
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode


class CapturingPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, JsonValue]]] = []

    async def publish(self, event_type: str, payload: dict[str, JsonValue]) -> None:
        self.events.append((event_type, payload))


def _state(research_plan: ResearchPlan | None = None) -> SupervisorState:
    return SupervisorState(
        user_request="Compare paper-a and paper-b on datasets and results",
        conversation_context="user: Compare paper-a and paper-b on datasets and results",
        research_plan=research_plan,
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


def _plan() -> ResearchPlan:
    return ResearchPlan(
        task_type=ResearchTaskType.COMPARISON,
        retrieval_strategy=RetrievalStrategy.MULTI_PAPER,
        answer_dimensions=["method", "results"],
        target_paper_count=2,
        comparison_targets=["paper-a", "paper-b"],
        requires_external_search=False,
        requires_local_corpus=True,
        stopping_criteria=["Both papers report every dimension"],
        rationale="A two-paper comparison over method and results",
    )


def _context(
    publisher: CapturingPublisher,
    *,
    paper_ids: tuple[str, ...] = (),
) -> AgentRunContext:
    return AgentRunContext(
        conversation_id=uuid4(),
        run_id=uuid4(),
        publisher=publisher,
        paper_ids=paper_ids,
        local_corpus_available=bool(paper_ids),
    )


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


@pytest.mark.asyncio
async def test_research_planner_publishes_an_explicit_plan() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).return_value = StructuredModelResult(
        value=_plan(),
        usage=ModelUsage(input_tokens=11, output_tokens=5, total_tokens=16),
    )
    publisher = CapturingPublisher()

    update = await ResearchPlannerNode(model)(
        _state(),
        Runtime(context=_context(publisher, paper_ids=("paper-a", "paper-b"))),
    )

    assert update["research_plan"] is not None
    assert update["research_plan"].task_type is ResearchTaskType.COMPARISON
    assert update["total_tokens"] == 16
    assert update["llm_calls"] == 1
    stages = [
        payload for event_type, payload in publisher.events if event_type == "stage.started"
    ]
    assert stages[0]["stage"] == "planner"
    assert stages[0]["indexed_paper_count"] == 2
    decisions = [
        payload
        for event_type, payload in publisher.events
        if event_type == "decision.recorded"
    ]
    assert decisions[0]["source"] == DecisionSource.MODEL.value
    assert decisions[0]["retrieval_strategy"] == RetrievalStrategy.MULTI_PAPER.value
    assert decisions[0]["answer_dimensions"] == ["method", "results"]
    plan_input = str(
        cast(AsyncMock, model.generate_structured).await_args.args[0][-1].content
    )
    assert "Compare paper-a and paper-b" in plan_input
    assert '"indexed_paper_ids": ["paper-a", "paper-b"]' in plan_input


@pytest.mark.asyncio
async def test_graph_runs_the_planner_before_the_supervisor_and_hands_it_the_plan() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).side_effect = [
        StructuredModelResult(value=_plan(), usage=ModelUsage(total_tokens=1)),
        StructuredModelResult(
            value=SupervisorDecision(
                assessment=DecisionAssessment(
                    observations=["The plan asks for a two-paper comparison"],
                    missing_information=["Comparison evidence"],
                    decision_summary="Read both named papers",
                ),
                task=WriterTask(objective="Answer from the plan", source_artifact_ids=[]),
            ),
            usage=ModelUsage(total_tokens=1),
        ),
    ]
    cast(AsyncMock, model.generate_text).return_value = TextModelResult(
        text="Done",
        usage=ModelUsage(total_tokens=1),
        message=AIMessage(content="Done"),
    )
    graph = SupervisorGraphBuilder(
        planner=ResearchPlannerNode(model),
        supervisor=SupervisorNode(model, max_steps=4),
        search=_unused_search_node(model),
        reader=ReaderAgentNode(model),
        analyst=AnalystAgentNode(model),
        writer=WriterAgentNode(
            model=model,
            recorder=cast(
                AgentExecutionRecorder,
                create_autospec(AgentExecutionRecorder, instance=True),
            ),
        ),
    ).build()
    publisher = CapturingPublisher()

    result = await graph.ainvoke(_state(), context=_context(publisher))

    assert result["research_plan"] is not None
    stage_names = [
        payload["stage"]
        for event_type, payload in publisher.events
        if event_type == "stage.started"
    ]
    assert stage_names[:2] == ["planner", "supervisor"]
    supervisor_input = str(
        cast(AsyncMock, model.generate_structured).await_args_list[1].args[0][-1].content
    )
    assert "Research plan (authoritative task strategy" in supervisor_input
    assert '"task_type":"comparison"' in supervisor_input.replace(", ", "")
    assert '"retrieval_strategy":"multi_paper"' in supervisor_input.replace(", ", "")


@pytest.mark.asyncio
async def test_graph_without_a_planner_starts_at_the_supervisor() -> None:
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_structured).return_value = StructuredModelResult(
        value=SupervisorDecision(
            assessment=DecisionAssessment(
                observations=["Nothing to read"],
                missing_information=[],
                decision_summary="Answer directly",
            ),
            task=WriterTask(objective="Answer directly", source_artifact_ids=[]),
        ),
        usage=ModelUsage(total_tokens=1),
    )
    cast(AsyncMock, model.generate_text).return_value = TextModelResult(
        text="Done",
        usage=ModelUsage(total_tokens=1),
        message=AIMessage(content="Done"),
    )
    graph = SupervisorGraphBuilder(
        supervisor=SupervisorNode(model, max_steps=4),
        search=_unused_search_node(model),
        reader=ReaderAgentNode(model),
        analyst=AnalystAgentNode(model),
        writer=WriterAgentNode(
            model=model,
            recorder=cast(
                AgentExecutionRecorder,
                create_autospec(AgentExecutionRecorder, instance=True),
            ),
        ),
    ).build()
    publisher = CapturingPublisher()

    await graph.ainvoke(_state(), context=_context(publisher))

    stage_names = [
        payload["stage"]
        for event_type, payload in publisher.events
        if event_type == "stage.started"
    ]
    assert stage_names == ["supervisor", "writer"]
    supervisor_input = str(
        cast(AsyncMock, model.generate_structured).await_args.args[0][-1].content
    )
    assert "not planned" in supervisor_input


def test_supervisor_keeps_only_local_targets_for_a_multi_paper_reader() -> None:
    task = ReaderTask(
        objective="Compare both papers",
        depth=ReaderDepth.DEEP,
        paper_ids=["paper-a", "external-paper"],
    )

    resolution = SupervisorNode._normalize_reader_scope(
        task,
        available_local_ids={"paper-a"},
        local_corpus_available=True,
        valid_search_ids=set(),
    )

    assert resolution.task is not None
    assert resolution.task.paper_ids == ["paper-a"]
    assert resolution.task.source_artifact_id is None
    rule = resolution.adjustments[-1].rule
    assert rule == "reader_local_targets_only"


def test_supervisor_keeps_every_local_paper_for_a_comparison() -> None:
    task = ReaderTask(
        objective="Compare three local papers",
        depth=ReaderDepth.DEEP,
        paper_ids=["paper-a", "paper-b", "paper-c"],
    )

    resolution = SupervisorNode._normalize_reader_scope(
        task,
        available_local_ids={"paper-a", "paper-b", "paper-c"},
        local_corpus_available=True,
        valid_search_ids=set(),
    )

    assert resolution.task is not None
    assert resolution.task.target_paper_ids == ["paper-a", "paper-b", "paper-c"]
    assert resolution.adjustments == []


def test_supervisor_limits_external_reading_to_one_paper() -> None:
    artifact_id = uuid4()
    task = ReaderTask(
        objective="Compare two external papers",
        depth=ReaderDepth.DEEP,
        paper_ids=["arxiv:1", "arxiv:2"],
        source_artifact_id=artifact_id,
    )

    resolution = SupervisorNode._normalize_reader_scope(
        task,
        available_local_ids=set(),
        local_corpus_available=False,
        valid_search_ids={artifact_id},
    )

    assert resolution.task is not None
    assert resolution.task.paper_id == "arxiv:1"
    assert resolution.task.paper_ids == []
    assert resolution.adjustments[-1].rule == "external_reader_single_paper_only"


def test_supervisor_requests_search_when_a_reader_has_no_readable_target() -> None:
    task = ReaderTask(
        objective="Read an unknown external paper",
        depth=ReaderDepth.DEEP,
        paper_id="arxiv:404",
    )

    resolution = SupervisorNode._normalize_reader_scope(
        task,
        available_local_ids=set(),
        local_corpus_available=False,
        valid_search_ids=set(),
    )

    assert resolution.task is None
    assert resolution.recovery_original == "arxiv:404"


def test_reader_task_scope_reaches_the_supervisor_decision_event() -> None:
    task = ReaderTask(
        objective="Compare both papers",
        depth=ReaderDepth.DEEP,
        paper_ids=["paper-a", "paper-b"],
    )

    payload = SupervisorNode._task_event_payload(task)

    assert payload["paper_ids"] == ["paper-a", "paper-b"]
    assert payload["retrieval_scope"] == "explicit_papers"


def test_search_task_payload_is_unchanged() -> None:
    task = SearchTask(objective="Find papers", query="retrieval augmented generation")

    payload = SupervisorNode._task_event_payload(task)

    assert payload["query"] == "retrieval augmented generation"
    assert payload["artifact_ids"] == []


def test_factory_wires_the_planner_node_into_the_supervisor_graph() -> None:
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    from app.infrastructure.agent.supervisor.factory import create_supervisor_agent

    model = ChatOpenAI(
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://localhost:1",
    )
    gateway = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    document_gateway = cast(
        PaperDocumentGateway,
        create_autospec(PaperDocumentGateway, instance=True),
    )
    store = cast(
        ConversationStore,
        create_autospec(ConversationStore, instance=True),
    )

    enabled = create_supervisor_agent(
        model=model,
        paper_search=gateway,
        paper_document=document_gateway,
        store=store,
        max_steps=4,
        search_max_iterations=1,
        reader_max_retrieval_rounds=1,
    )
    disabled = create_supervisor_agent(
        model=model,
        paper_search=gateway,
        paper_document=document_gateway,
        store=store,
        max_steps=4,
        search_max_iterations=1,
        reader_max_retrieval_rounds=1,
        enable_research_planner=False,
    )

    assert "planner" in enabled._graph.get_graph().nodes  # noqa: SLF001
    assert "planner" not in disabled._graph.get_graph().nodes  # noqa: SLF001
