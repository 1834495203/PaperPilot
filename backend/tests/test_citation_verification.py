"""Citation traceability and citation-support verification.

Coming from the right paper is not the same as supporting the claim, so the Writer
resolves evidence through artifact source relationships and then checks each
citation against the evidence it points at.
"""

import json
from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.ports import EventPublisher
from app.domain.rag import Evidence
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.supervisor.model_gateway import (
    AgentModelGateway,
    ModelUsage,
    StructuredModelResult,
    TextModelResult,
)
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    AgentName,
    AnalystAgentSummary,
    ArtifactKind,
    CitationSupportAssessment,
    CitationSupportItem,
    DecisionAssessment,
    DecisionSource,
    ReaderAgentSummary,
    SupervisorDecision,
    WriterTask,
)
from app.infrastructure.agent.supervisor.state import SupervisorState
from app.infrastructure.agent.supervisor.support import resolve_evidence
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode

EVIDENCE_ID = "E-1234567890ab"


class CapturingPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, JsonValue]]] = []

    async def publish(self, event_type: str, payload: dict[str, JsonValue]) -> None:
        self.events.append((event_type, payload))


def _evidence(evidence_id: str = EVIDENCE_ID) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        paper_id="paper-x",
        paper_title="A Test Paper",
        chunk_id="paper-x:chunk:1",
        section_path=["1 Method"],
        page_start=4,
        page_end=4,
        raw_text="self-attention improves parallelization",
        evidence_text="self-attention improves parallelization",
        retrieval_score=0.9,
        spans=[],
    )


def _reader_artifact() -> AgentArtifact:
    return AgentArtifact(
        title="Reading report",
        supervisor_summary=ReaderAgentSummary(
            summary="Read the paper",
            paper_or_material="paper-x",
            objective_satisfied=True,
            answered_points=["method"],
            blocking_gaps=[],
            evidence_scope="Full text",
            limitations=[],
            pdf_truncated=False,
        ),
        content=json.dumps(
            {
                "reading_report": {"paper_or_material": "paper-x"},
                "evidence_library": {
                    "objective": "read",
                    "evidence": [_evidence().model_dump(mode="json")],
                },
            }
        ),
    )


def _analyst_artifact(source_ids: list[object]) -> AgentArtifact:
    return AgentArtifact(
        title="Analysis report",
        supervisor_summary=AnalystAgentSummary(
            summary="Compared the supplied report",
            finding_count=1,
            research_gap_count=0,
            novelty_assessment="Not assessed",
            research_gaps=[],
            unresolved_questions=[],
        ),
        content=json.dumps({"analysis_report": {"analysis_type": "comparison"}}),
        source_artifact_ids=source_ids,  # type: ignore[arg-type]
    )


def _state(artifacts: list[AgentArtifact], source_ids: list[object]) -> SupervisorState:
    return SupervisorState(
        user_request="How does attention help?",
        conversation_context="user: How does attention help?",
        research_plan=None,
        artifacts=artifacts,
        completed_steps=[],
        decision=SupervisorDecision(
            assessment=DecisionAssessment(
                observations=["Analysis is available"],
                missing_information=[],
                decision_summary="Answer from the analysis",
            ),
            task=WriterTask(
                objective="Answer the question",
                source_artifact_ids=source_ids,  # type: ignore[arg-type]
            ),
        ),
        step_count=0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        llm_calls=0,
        tool_calls=0,
    )


def test_evidence_resolves_through_artifact_sources() -> None:
    reader = _reader_artifact()
    analyst = _analyst_artifact([reader.id])

    resolved = resolve_evidence([reader, analyst], [analyst.id])

    assert list(resolved) == [EVIDENCE_ID]
    assert resolved[EVIDENCE_ID].paper_title == "A Test Paper"


def test_evidence_resolution_tolerates_missing_sources() -> None:
    analyst = _analyst_artifact([uuid4()])

    assert resolve_evidence([analyst], [analyst.id, uuid4()]) == {}


@pytest.mark.asyncio
async def test_writer_can_cite_evidence_reached_through_an_analyst_report() -> None:
    reader = _reader_artifact()
    analyst = _analyst_artifact([reader.id])
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_text).return_value = TextModelResult(
        text=f"Attention improves parallelization [{EVIDENCE_ID}].",
        usage=ModelUsage(total_tokens=3),
        message=AIMessage(content="answer"),
    )
    cast(AsyncMock, model.generate_structured).return_value = StructuredModelResult(
        value=CitationSupportAssessment(
            assessments=[
                CitationSupportItem(
                    evidence_id=EVIDENCE_ID,
                    supported=True,
                    reason="The evidence states exactly this",
                )
            ]
        ),
        usage=ModelUsage(total_tokens=2),
    )
    recorder = cast(
        AgentExecutionRecorder,
        create_autospec(AgentExecutionRecorder, instance=True),
    )
    cast(AsyncMock, recorder.record_assistant_message).return_value = uuid4()
    publisher = CapturingPublisher()

    await WriterAgentNode(model=model, recorder=recorder)(
        _state([reader, analyst], [analyst.id]),
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=publisher,
            )
        ),
    )

    kwargs = cast(AsyncMock, recorder.record_assistant_message).await_args.kwargs
    assert [item["evidence_id"] for item in kwargs["citations"]] == [EVIDENCE_ID]
    verification = kwargs["citation_verification"]
    assert verification["issues"] == []
    assert verification["supported"] == [EVIDENCE_ID]
    completed = [
        payload
        for event_type, payload in publisher.events
        if event_type == EventType.MESSAGE_COMPLETED.value
    ]
    assert completed[0]["citation_verification"]["checked"] == 1
    stages = [
        payload["stage"]
        for event_type, payload in publisher.events
        if event_type == EventType.STAGE_STARTED.value
    ]
    assert "writer.citation_check" in stages


@pytest.mark.asyncio
async def test_writer_reports_an_unsupported_citation() -> None:
    reader = _reader_artifact()
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_text).return_value = TextModelResult(
        text=f"Attention removes the need for recurrence [{EVIDENCE_ID}].",
        usage=ModelUsage(total_tokens=3),
        message=AIMessage(content="answer"),
    )
    cast(AsyncMock, model.generate_structured).return_value = StructuredModelResult(
        value=CitationSupportAssessment(
            assessments=[
                CitationSupportItem(
                    evidence_id=EVIDENCE_ID,
                    supported=False,
                    reason="The evidence only mentions parallelization",
                )
            ]
        ),
        usage=ModelUsage(total_tokens=2),
    )
    recorder = cast(
        AgentExecutionRecorder,
        create_autospec(AgentExecutionRecorder, instance=True),
    )
    cast(AsyncMock, recorder.record_assistant_message).return_value = uuid4()

    await WriterAgentNode(model=model, recorder=recorder)(
        _state([reader], [reader.id]),
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
            )
        ),
    )

    verification = cast(AsyncMock, recorder.record_assistant_message).await_args.kwargs[
        "citation_verification"
    ]
    assert verification["supported"] == []
    assert [item["kind"] for item in verification["issues"]] == ["unsupported_claim"]
    assert "parallelization" in verification["issues"][0]["detail"]


@pytest.mark.asyncio
async def test_verification_failure_does_not_fail_the_answer() -> None:
    reader = _reader_artifact()
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_text).return_value = TextModelResult(
        text=f"Attention helps [{EVIDENCE_ID}].",
        usage=ModelUsage(total_tokens=3),
        message=AIMessage(content="answer"),
    )
    cast(AsyncMock, model.generate_structured).side_effect = RuntimeError("judge offline")
    recorder = cast(
        AgentExecutionRecorder,
        create_autospec(AgentExecutionRecorder, instance=True),
    )
    cast(AsyncMock, recorder.record_assistant_message).return_value = uuid4()

    await WriterAgentNode(model=model, recorder=recorder)(
        _state([reader], [reader.id]),
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
            )
        ),
    )

    verification = cast(AsyncMock, recorder.record_assistant_message).await_args.kwargs[
        "citation_verification"
    ]
    assert verification["verification_error"] == "judge offline"
    assert verification["checked"] == 1


@pytest.mark.asyncio
async def test_unknown_citation_ids_are_recorded_as_issues() -> None:
    reader = _reader_artifact()
    model = cast(AgentModelGateway, create_autospec(AgentModelGateway, instance=True))
    cast(AsyncMock, model.generate_text).return_value = TextModelResult(
        text="Attention helps [E-deadbeef0000].",
        usage=ModelUsage(total_tokens=3),
        message=AIMessage(content="answer"),
    )
    recorder = cast(
        AgentExecutionRecorder,
        create_autospec(AgentExecutionRecorder, instance=True),
    )
    cast(AsyncMock, recorder.record_assistant_message).return_value = uuid4()

    await WriterAgentNode(model=model, recorder=recorder)(
        _state([reader], [reader.id]),
        Runtime(
            context=AgentRunContext(
                conversation_id=uuid4(),
                run_id=uuid4(),
                publisher=CapturingPublisher(),
            )
        ),
    )

    kwargs = cast(AsyncMock, recorder.record_assistant_message).await_args.kwargs
    assert kwargs["citations"] == []
    assert [item["kind"] for item in kwargs["citation_verification"]["issues"]] == [
        "unknown_evidence_id"
    ]
    # No resolvable citation means nothing to verify with the model.
    cast(AsyncMock, model.generate_structured).assert_not_awaited()


def test_artifacts_expose_their_kind_for_recursive_resolution() -> None:
    analyst = _analyst_artifact([])

    assert analyst.kind is ArtifactKind.ANALYSIS_REPORT
    assert AnalystAgentSummary.model_fields["kind"].default is ArtifactKind.ANALYSIS_REPORT
    assert AgentName.WRITER.value == "writer"
    assert DecisionSource.MODEL.value == "model"
