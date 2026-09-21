"""Supervisor-facing Reader agent.

The Reader's own work lives in the ``reader`` package: state contract, material
acquisition, evidence rules, prompt assembly and reporting. This module is the
adapter that validates the assigned task, runs the subgraph and reports its
result back into the supervisor state.
"""

from typing import cast

from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.application.paper_library import PaperLibraryService
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.enums import EventType
from app.domain.ports import PaperDocumentGateway
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway, ModelUsage
from app.infrastructure.agent.supervisor.models import (
    AgentName,
    CompletedStep,
    DecisionSource,
    ReaderTask,
)
from app.infrastructure.agent.supervisor.reader.evidence import (
    build_evidence_library,
    merge_coverage,
)
from app.infrastructure.agent.supervisor.reader.graph import ReaderAgentGraph
from app.infrastructure.agent.supervisor.reader.materials import select_pdf_candidate
from app.infrastructure.agent.supervisor.reader.reporting import (
    build_reader_artifact,
    build_reader_decision_payload,
    build_reader_outcome,
)
from app.infrastructure.agent.supervisor.reader.state import ReaderState
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    select_artifacts,
    with_usage,
)


class ReaderAgentNode:
    """Supervisor-facing adapter around the independent Reader subgraph."""

    def __init__(
        self,
        model: AgentModelGateway,
        *,
        document_gateway: PaperDocumentGateway | None = None,
        paper_retriever: TreeRagRetriever | None = None,
        paper_library: PaperLibraryService | None = None,
        recorder: AgentExecutionRecorder | None = None,
        max_retrieval_rounds: int = 2,
    ) -> None:
        self._graph = ReaderAgentGraph(
            model,
            document_gateway=document_gateway,
            paper_retriever=paper_retriever,
            paper_library=paper_library,
            recorder=recorder,
            max_retrieval_rounds=max_retrieval_rounds,
        )

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        task = self._reader_task(state)
        targets = task.target_paper_ids
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "reader",
                "stage": "reader",
                "summary": "Reader Agent 子图开始执行",
                "objective": task.objective,
                "reader_depth": task.depth.value,
                "paper_ids": cast(JsonValue, targets),
                "retrieval_scope": (
                    "all_local_papers" if not targets else "explicit_papers"
                ),
                "research_task_type": (
                    state["research_plan"].task_type.value
                    if state["research_plan"] is not None
                    else None
                ),
            },
        )
        selected = select_artifacts(
            state["artifacts"],
            [task.source_artifact_id] if task.source_artifact_id is not None else [],
        )
        is_local_paper = (
            all(paper_id in context.paper_ids for paper_id in targets)
            if targets
            else context.local_corpus_available or bool(context.paper_ids)
        )
        result = await self._graph.run(
            ReaderState(
                task=task,
                research_plan=state["research_plan"],
                is_local_paper=is_local_paper,
                local_metadata=None,
                pdf_candidate=(
                    select_pdf_candidate(selected, task.paper_id)
                    if task.paper_id is not None
                    else None
                ),
                pdf_document=None,
                material_error=None,
                plan=None,
                assessment=None,
                coverage_judgments=[],
                retrieval_reports=[],
                retrieval_attempts=0,
                attempted_queries=[],
                final_report=None,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                llm_calls=0,
                tool_calls=0,
            ),
            context=context,
        )
        report = result["final_report"]
        if report is None:
            raise RuntimeError("Reader subgraph completed without a reading report")
        retrieval_reports = result["retrieval_reports"]
        pdf_document = result["pdf_document"]
        library = build_evidence_library(
            task,
            retrieval_reports,
            pdf_document,
            result["pdf_candidate"],
            result["local_metadata"],
        )
        coverage = merge_coverage(
            retrieval_reports,
            result["plan"],
            result["coverage_judgments"],
            library,
        )
        artifact = build_reader_artifact(
            task=task,
            report=report,
            library=library,
            coverage=coverage,
            retrieval_reports=retrieval_reports,
            pdf_document=pdf_document,
            material_error=result["material_error"],
            attempted_queries=result["attempted_queries"],
            retrieval_rounds=result["retrieval_attempts"],
            source_artifact_ids=[item.id for item in selected],
        )
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            build_reader_decision_payload(
                task=task,
                artifact=artifact,
                report=report,
                coverage=coverage,
                pdf_document=pdf_document,
                source_count=len(selected),
            ),
        )
        update: SupervisorStateUpdate = {
            "artifacts": [*state["artifacts"], artifact],
            "completed_steps": [
                *state["completed_steps"],
                CompletedStep(
                    agent=AgentName.READER,
                    objective=task.objective,
                    artifact_id=artifact.id,
                ),
            ],
            "reader_outcome": build_reader_outcome(
                task=task,
                artifact=artifact,
                report=report,
                assessment=result["assessment"],
            ),
            "decision": None,
            **with_usage(
                state,
                ModelUsage(
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    total_tokens=result["total_tokens"],
                ),
                llm_calls=result["llm_calls"],
                tool_calls=result["tool_calls"],
            ),
        }
        await publish_metrics(context, update)
        return update

    @staticmethod
    def _reader_task(state: SupervisorState) -> ReaderTask:
        decision = state["decision"]
        if decision is None or not isinstance(decision.task, ReaderTask):
            raise ValueError("Reader Agent requires a reader decision")
        return decision.task
