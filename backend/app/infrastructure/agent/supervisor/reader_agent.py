import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from app.application.agent import AgentRunContext
from app.application.paper_library import PaperLibraryService, PaperNotFoundError
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.enums import EventType
from app.domain.papers import Paper, PdfDocument
from app.domain.ports import PaperDocumentGateway
from app.domain.rag import (
    CoverageCell,
    CoverageMatrix,
    CoverageStatus,
    Evidence,
    EvidenceLibrary,
    RetrievalMode,
    RetrievalQuery,
    RetrievalStrategy,
    TreeRetrievalReport,
)
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway, ModelUsage
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    AgentName,
    ArtifactKind,
    CompletedStep,
    CoverageGapJudgment,
    DecisionAssessment,
    DecisionSource,
    ReaderAgentSummary,
    ReaderDepth,
    ReaderTask,
    ReadingEvidenceAssessment,
    ReadingPlan,
    ReadingReport,
    ReadingSubQuestion,
    ResearchPlan,
    SearchReport,
    SupervisorDecision,
    WriterTask,
)
from app.infrastructure.agent.supervisor.prompts import (
    READER_EVIDENCE_PROMPT,
    READER_PLANNING_PROMPT,
    READER_PROMPT,
)
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    select_artifacts,
    with_usage,
)


@dataclass(frozen=True, slots=True)
class PdfCandidate:
    paper: Paper
    source_artifact_id: str


class ReaderState(TypedDict):
    task: ReaderTask
    research_plan: ResearchPlan | None
    selected_artifacts: list[AgentArtifact]
    is_local_paper: bool
    local_metadata: dict[str, JsonValue] | None
    pdf_candidate: PdfCandidate | None
    pdf_document: PdfDocument | None
    material_error: str | None
    plan: ReadingPlan | None
    assessment: ReadingEvidenceAssessment | None
    coverage_judgments: list[CoverageGapJudgment]
    retrieval_reports: list[TreeRetrievalReport]
    retrieval_attempts: int
    attempted_queries: list[str]
    final_report: ReadingReport | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int


class ReaderStateUpdate(TypedDict, total=False):
    local_metadata: dict[str, JsonValue] | None
    pdf_document: PdfDocument | None
    material_error: str | None
    plan: ReadingPlan | None
    assessment: ReadingEvidenceAssessment | None
    coverage_judgments: list[CoverageGapJudgment]
    retrieval_reports: list[TreeRetrievalReport]
    retrieval_attempts: int
    attempted_queries: list[str]
    final_report: ReadingReport | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int


PlanRoute = Literal["retrieve", "fetch", "assess"]
EvidenceRoute = Literal["retrieve_gaps", "synthesize"]

class ReaderAgentGraph:
    """Plan-retrieve-assess-repair LangGraph subgraph for local or external evidence.

    Nodes: prepare -> plan -> retrieve -> assess -> (retrieve_gaps -> assess)* -> synthesize.
    Planning, evidence judging and coverage verdicts are model decisions; only the
    quick path and the gap-loop budget are policy.
    """

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
        if max_retrieval_rounds < 1:
            raise ValueError("max_retrieval_rounds must be positive")
        self._model = model
        self._document_gateway = document_gateway
        self._paper_retriever = paper_retriever
        self._paper_library = paper_library
        self._recorder = recorder
        self._max_retrieval_rounds = max_retrieval_rounds

        builder = StateGraph(ReaderState, context_schema=AgentRunContext)
        builder.add_node("prepare", self._prepare)
        builder.add_node("plan", self._plan)
        builder.add_node("fetch", self._fetch)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("assess", self._assess)
        builder.add_node("retrieve_gaps", self._retrieve_gaps)
        builder.add_node("synthesize", self._synthesize)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "plan")
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"retrieve": "retrieve", "fetch": "fetch", "assess": "assess"},
        )
        builder.add_edge("fetch", "assess")
        builder.add_edge("retrieve", "assess")
        builder.add_conditional_edges(
            "assess",
            self._route_after_assessment,
            {"retrieve_gaps": "retrieve_gaps", "synthesize": "synthesize"},
        )
        builder.add_edge("retrieve_gaps", "assess")
        builder.add_edge("synthesize", END)
        self._graph = builder.compile()

    async def run(
        self,
        initial_state: ReaderState,
        *,
        context: AgentRunContext,
    ) -> ReaderState:
        result = await self._graph.ainvoke(initial_state, context=context)
        return cast(ReaderState, result)

    async def _prepare(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        del runtime
        task = state["task"]
        local_metadata: dict[str, JsonValue] | None = None
        material_error: str | None = None
        targets = task.target_paper_ids
        if state["is_local_paper"]:
            if self._paper_library is not None and targets:
                local_metadata = {}
                for paper_id in targets:
                    try:
                        paper = await self._paper_library.get_paper(paper_id)
                    except PaperNotFoundError:
                        material_error = "Local paper metadata was not found"
                        continue
                    local_metadata[paper_id] = cast(
                        JsonValue,
                        paper.metadata.model_dump(mode="json"),
                    )
        elif len(targets) > 1:
            material_error = (
                "External reading covers exactly one paper; multiple external paper IDs "
                "were requested"
            )
        else:
            candidate = state["pdf_candidate"]
            if candidate is None:
                material_error = (
                    "The requested paper ID was not found in the selected search artifact"
                )
            else:
                local_metadata = cast(dict[str, JsonValue], candidate.paper.model_dump(mode="json"))
        return {
            "local_metadata": local_metadata,
            "material_error": material_error,
        }

    async def _plan(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        """Model-decided retrieval strategy and sub-question decomposition."""

        if state["task"].depth is ReaderDepth.QUICK:
            return await self._quick_plan(state, runtime)
        context = runtime.context
        task = state["task"]
        targets = task.target_paper_ids
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "reader",
                "stage": "reader.plan",
                "summary": "Reader 正在制定论文库检索计划",
                "paper_ids": cast(JsonValue, targets),
                "retrieval_scope": (
                    "all_local_papers" if not targets else "explicit_papers"
                ),
                "objective": task.objective,
            },
        )
        result = await self._model.generate_structured(
            [
                SystemMessage(content=READER_PLANNING_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "paper_ids": targets,
                            "retrieval_scope": (
                                "all_local_papers" if not targets else "explicit_papers"
                            ),
                            "retrieval_modes": [item.value for item in RetrievalMode],
                            "paper_metadata": state["local_metadata"],
                            "research_plan": (
                                state["research_plan"].model_dump(mode="json")
                                if state["research_plan"] is not None
                                else None
                            ),
                            "reading_objective": task.objective,
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            ReadingPlan,
        )
        plan = result.value
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "reader",
                "stage": "reader.plan",
                "summary": plan.rationale,
                "needs_retrieval": plan.needs_retrieval,
                "query": plan.query,
                "retrieval_mode": plan.mode.value if plan.mode is not None else None,
                "retrieval_strategy": (
                    plan.strategy.value if plan.strategy is not None else None
                ),
                "coverage_dimensions": cast(JsonValue, plan.coverage_dimensions),
                "sub_question_count": len(plan.sub_questions),
                "sub_questions": cast(
                    JsonValue,
                    [item.model_dump(mode="json") for item in plan.sub_questions],
                ),
                "evidence_requirements": cast(JsonValue, plan.evidence_requirements),
            },
        )
        return {"plan": plan, **self._usage_update(state, result.usage)}

    async def _quick_plan(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        """Narrow questions skip the planning model and keep one exact query."""

        task = state["task"]
        targets = task.target_paper_ids
        needs_retrieval = (
            state["is_local_paper"] and self._paper_retriever is not None
        ) or (not state["is_local_paper"] and self._document_gateway is not None)
        plan = ReadingPlan(
            needs_retrieval=needs_retrieval,
            query=task.objective if needs_retrieval else None,
            mode=RetrievalMode.METHOD if needs_retrieval else None,
            strategy=(
                RetrievalStrategy.MULTI_PAPER
                if len(targets) > 1
                else RetrievalStrategy.SINGLE_PAPER
            ),
            coverage_dimensions=(
                list(state["research_plan"].answer_dimensions)
                if state["research_plan"] is not None
                else []
            ),
            sub_questions=(
                [
                    ReadingSubQuestion(
                        question=task.objective,
                        retrieval_query=task.objective,
                        mode=RetrievalMode.METHOD,
                        paper_ids=targets,
                    )
                ]
                if needs_retrieval
                else []
            ),
            evidence_requirements=[task.objective],
            rationale="Quick path uses the exact user-scoped objective for one retrieval",
        )
        await runtime.context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.POLICY.value,
                "actor": "reader",
                "stage": "reader.quick_plan",
                "summary": plan.rationale,
                "needs_retrieval": plan.needs_retrieval,
                "query": plan.query,
                "retrieval_mode": plan.mode.value if plan.mode is not None else None,
                "retrieval_strategy": plan.strategy.value if plan.strategy else None,
            },
        )
        return {"plan": plan}

    async def _fetch(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        candidate = state["pdf_candidate"]
        if candidate is None:
            return {"material_error": "No external PDF candidate is available"}
        if self._document_gateway is None:
            return {"material_error": "PDF document gateway is not configured"}
        document, error = await self._fetch_pdf(runtime.context, candidate)
        return {
            "pdf_document": document,
            "material_error": error,
            "tool_calls": state["tool_calls"] + 1,
        }

    async def _retrieve(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        plan = state["plan"]
        if plan is None:
            raise ValueError("Reader retrieval requires a reading plan")
        sub_questions = plan.retrieval_queries()
        if not sub_questions:
            raise ValueError("Reader retrieval requires at least one sub-question")
        return await self._run_retrieval_round(
            state,
            runtime.context,
            sub_questions=sub_questions,
            strategy=plan.strategy,
            stage="reader.retrieve",
        )

    async def _retrieve_gaps(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        """Second node in the loop: targeted retrieval for judged evidence gaps."""

        assessment = state["assessment"]
        plan = state["plan"]
        if assessment is None or plan is None:
            raise ValueError("Reader gap retrieval requires an evidence assessment")
        gaps = assessment.gap_queries()
        if not gaps:
            raise ValueError("Reader gap retrieval requires at least one gap query")
        await runtime.context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "reader",
                "stage": "reader.retrieve_gaps",
                "summary": "Reader 正在按覆盖缺口定向补检",
                "gap_count": len(gaps),
                "gaps": cast(
                    JsonValue,
                    [item.model_dump(mode="json") for item in gaps],
                ),
            },
        )
        return await self._run_retrieval_round(
            state,
            runtime.context,
            sub_questions=[
                ReadingSubQuestion(
                    question=item.query,
                    retrieval_query=item.query,
                    mode=item.mode,
                    paper_ids=item.paper_ids,
                    dimension=item.dimension,
                )
                for item in gaps
            ],
            strategy=plan.strategy,
            stage="reader.retrieve_gaps",
        )

    async def _run_retrieval_round(
        self,
        state: ReaderState,
        context: AgentRunContext,
        *,
        sub_questions: list[ReadingSubQuestion],
        strategy: RetrievalStrategy | None,
        stage: str,
    ) -> ReaderStateUpdate:
        at_capacity = state["retrieval_attempts"] >= self._max_retrieval_rounds
        truncated = list(sub_questions)
        if at_capacity:
            truncated = truncated[:1]
        report, error = await self._retrieve_indexed_paper(
            context,
            paper_ids=state["task"].target_paper_ids,
            queries=truncated,
            strategy=strategy,
            stage=stage,
        )
        reports = list(state["retrieval_reports"])
        if report is not None:
            reports.append(report)
        return {
            "retrieval_reports": reports,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
            "attempted_queries": [
                *state["attempted_queries"],
                *(item.retrieval_query for item in truncated),
            ],
            "material_error": error,
            "assessment": None,
            "tool_calls": state["tool_calls"] + 1,
        }

    async def _assess(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        context = runtime.context
        plan = state["plan"]
        if plan is None:
            raise ValueError("Reader evidence assessment requires a reading plan")
        coverage = self._coverage_matrix(state)
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "reader",
                "stage": "reader.assess",
                "summary": "Judge 正在按用户原问题检查证据与覆盖充分性",
                "retrieval_round": len(state["retrieval_reports"]),
                "candidate_ratio": coverage.candidate_ratio if coverage is not None else 0.0,
                "candidate_cell_count": (
                    coverage.candidate_cell_count if coverage is not None else 0
                ),
                "unresolved_cell_count": (
                    coverage.unresolved_cell_count if coverage is not None else 0
                ),
            },
        )
        result = await self._model.generate_structured(
            [
                SystemMessage(content=READER_EVIDENCE_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "reading_objective": state["task"].objective,
                            "reader_depth": state["task"].depth.value,
                            "retrieval_can_retry": (
                                state["task"].depth is ReaderDepth.DEEP
                                and state["is_local_paper"]
                                and self._paper_retriever is not None
                                and state["retrieval_attempts"] < self._max_retrieval_rounds
                            ),
                            "retrieval_strategy": (
                                plan.strategy.value if plan.strategy is not None else None
                            ),
                            "reading_plan": plan.model_dump(mode="json"),
                            "coverage_matrix": (
                                coverage.model_dump(mode="json")
                                if coverage is not None
                                else None
                            ),
                            "unresolved_cells": cast(
                                JsonValue,
                                [
                                    cell.model_dump(mode="json")
                                    for cell in (
                                        coverage.unresolved_cells()
                                        if coverage is not None
                                        else []
                                    )
                                ],
                            ),
                            "available_material": self._reading_material(state),
                            "material_error": state["material_error"],
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            ReadingEvidenceAssessment,
        )
        assessment = result.value
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "reader",
                "stage": "reader.assess",
                "summary": assessment.coverage_summary,
                "evidence_sufficient": assessment.evidence_sufficient,
                "covered_requirements": cast(JsonValue, assessment.covered_requirements),
                "missing_requirements": cast(JsonValue, assessment.missing_requirements),
                "retry_recommended": assessment.retry_recommended,
                "next_queries": cast(
                    JsonValue,
                    [item.model_dump(mode="json") for item in assessment.next_queries],
                ),
                "coverage_judgments": cast(
                    JsonValue,
                    [
                        item.model_dump(mode="json")
                        for item in assessment.coverage_judgments
                    ],
                ),
                "next_query": assessment.next_query,
                "next_mode": (
                    assessment.next_mode.value if assessment.next_mode is not None else None
                ),
            },
        )
        return {
            "assessment": assessment,
            "coverage_judgments": self._merge_judgments(
                state["coverage_judgments"],
                assessment.coverage_judgments,
            ),
            **self._usage_update(state, result.usage),
        }

    async def _synthesize(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        del runtime
        task = state["task"]
        coverage = self._coverage_matrix(state)
        scope = (
            "all locally indexed papers"
            if not task.target_paper_ids
            else task.target_paper_ids
        )
        prompt = (
            f"Assigned evidence-reading task:\n{task.objective}\n\n"
            f"Reader depth:\n{task.depth.value}\n\n"
            f"Retrieval scope:\n{scope}\n\n"
            f"Available material:\n{self._reading_material(state)}\n\n"
            f"Coverage matrix:\n"
            f"{coverage.model_dump_json() if coverage is not None else 'not available'}\n\n"
            "Analyze only the supplied evidence. Do not evaluate the overall workflow. "
            "Report each paper and dimension cell that stayed uncovered, and state plainly when "
            "the paper does not address a dimension instead of treating it as missing evidence."
        )
        result = await self._model.generate_structured(
            [SystemMessage(content=READER_PROMPT), HumanMessage(content=prompt)],
            ReadingReport,
        )
        report = self._validate_evidence_references(
            result.value,
            self._evidence_library(state),
        )
        return {
            "final_report": report,
            **self._usage_update(state, result.usage),
        }

    def _route_after_plan(self, state: ReaderState) -> PlanRoute:
        plan = state["plan"]
        if plan is None:
            raise ValueError("Reader planning node did not produce a plan")
        if not plan.needs_retrieval:
            return "assess"
        if state["is_local_paper"] and self._paper_retriever is not None:
            return "retrieve"
        if not state["is_local_paper"] and self._document_gateway is not None:
            return "fetch"
        return "assess"

    def _route_after_assessment(self, state: ReaderState) -> EvidenceRoute:
        assessment = state["assessment"]
        if assessment is None:
            raise ValueError("Reader assessment node did not produce a decision")
        if (
            state["task"].depth is ReaderDepth.DEEP
            and state["is_local_paper"]
            and self._paper_retriever is not None
            and not assessment.evidence_sufficient
            and assessment.retry_recommended
            and bool(assessment.gap_queries())
            and state["retrieval_attempts"] < self._max_retrieval_rounds
        ):
            return "retrieve_gaps"
        return "synthesize"

    @staticmethod
    def _merge_judgments(
        existing: list[CoverageGapJudgment],
        incoming: list[CoverageGapJudgment],
    ) -> list[CoverageGapJudgment]:
        """Keep every judged cell, letting a later round revise an earlier verdict."""

        merged: dict[tuple[str, str], CoverageGapJudgment] = {
            (item.paper_id, item.dimension): item for item in existing
        }
        for item in incoming:
            merged[(item.paper_id, item.dimension)] = item
        return list(merged.values())

    @staticmethod
    def _usage_update(state: ReaderState, usage: ModelUsage) -> ReaderStateUpdate:        return {
            "input_tokens": state["input_tokens"] + usage.input_tokens,
            "output_tokens": state["output_tokens"] + usage.output_tokens,
            "total_tokens": state["total_tokens"] + usage.total_tokens,
            "llm_calls": state["llm_calls"] + 1,
        }

    @staticmethod
    def _reading_material(state: ReaderState) -> str:
        evidence_library = ReaderAgentGraph._evidence_library(state)
        if state["retrieval_reports"]:
            return json.dumps(
                {
                    "paper_ids": state["task"].target_paper_ids,
                    "paper_metadata": state["local_metadata"],
                    "reading_plan": (
                        state["plan"].model_dump(mode="json") if state["plan"] is not None else None
                    ),
                    "evidence_assessment": (
                        state["assessment"].model_dump(mode="json")
                        if state["assessment"] is not None
                        else None
                    ),
                    "retrieval_rounds": [
                        {
                            "queries": [
                                item.model_dump(mode="json") for item in report.queries
                            ],
                            "mode": report.mode.value,
                            "strategy": report.strategy.value,
                            "candidate_paper_ids": report.candidate_paper_ids,
                            "missing_paper_ids": report.missing_paper_ids,
                            "coverage": (
                                report.coverage.model_dump(mode="json")
                                if report.coverage is not None
                                else None
                            ),
                            "initial_hit_count": report.initial_hit_count,
                            "expanded_candidate_count": report.expanded_candidate_count,
                            "deduplicated_candidate_count": (
                                report.deduplicated_candidate_count
                            ),
                            "mmr_candidate_count": report.mmr_candidate_count,
                            "keyword_candidate_count": report.keyword_candidate_count,
                            "reranker_name": report.reranker_name,
                            "reranker_applied": report.reranker_applied,
                            "reranker_error": report.reranker_error,
                        }
                        for report in state["retrieval_reports"]
                    ],
                    "evidence_library": evidence_library.model_dump(mode="json"),
                    "instruction": (
                        "paper_metadata maps each known paper_id to its authoritative "
                        "bibliographic metadata. Use only Evidence Library entries for "
                        "substantive claims and copy their exact evidence IDs into report "
                        "evidence entries."
                    ),
                },
                ensure_ascii=False,
            )
        document = state["pdf_document"]
        candidate = state["pdf_candidate"]
        if document is not None:
            return json.dumps(
                {
                    "paper": candidate.paper.model_dump(mode="json") if candidate else None,
                    "pdf_extraction": {
                        "source_url": str(document.source_url),
                        "page_count": document.page_count,
                        "extracted_pages": document.extracted_pages,
                        "extracted_characters": document.extracted_characters,
                        "truncated": document.truncated,
                        "warnings": document.extraction_warnings,
                    },
                    "evidence_library": evidence_library.model_dump(mode="json"),
                },
                ensure_ascii=False,
            )
        material = json.dumps(
            {
                "paper_metadata": state["local_metadata"],
                "paper": candidate.paper.model_dump(mode="json") if candidate else None,
                "evidence_library": evidence_library.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        return (
            f"{material}\n\nMaterial retrieval failed: {state['material_error']}"
            if state["material_error"]
            else material
        )

    @classmethod
    def _coverage_matrix(cls, state: ReaderState) -> CoverageMatrix | None:
        """Merge every retrieval round with the judge verdicts into one matrix."""

        reports = state["retrieval_reports"]
        if not reports:
            return None
        plan = state["plan"]
        paper_ids: list[str] = []
        dimensions: list[str] = []
        candidates: dict[tuple[str, str], list[str]] = {}
        for report in reports:
            matrix = report.coverage
            if matrix is not None:
                for dimension in matrix.dimensions:
                    if dimension not in dimensions:
                        dimensions.append(dimension)
                for cell in matrix.cells:
                    if cell.paper_id not in paper_ids:
                        paper_ids.append(cell.paper_id)
                    if cell.evidence_ids:
                        existing = candidates.get((cell.paper_id, cell.dimension), [])
                        candidates[(cell.paper_id, cell.dimension)] = [
                            *existing,
                            *(item for item in cell.evidence_ids if item not in existing),
                        ]
            for paper_id in report.missing_paper_ids:
                if paper_id not in paper_ids:
                    paper_ids.append(paper_id)
        if not paper_ids:
            return None
        preferred = list(plan.coverage_dimensions) if plan is not None else []
        matrix = CoverageMatrix.build(
            paper_ids=paper_ids,
            dimensions=preferred or dimensions,
        )
        for (paper_id, dimension), evidence_ids in candidates.items():
            if matrix.cell_for(paper_id, dimension) is None:
                continue
            matrix = matrix.with_cell(
                CoverageCell(
                    paper_id=paper_id,
                    dimension=dimension,
                    status=CoverageStatus.CANDIDATE,
                    evidence_ids=evidence_ids,
                )
            )
        return cls._apply_judgments(
            matrix,
            state["coverage_judgments"],
            library=cls._evidence_library(state),
        )

    @classmethod
    def _apply_judgments(
        cls,
        matrix: CoverageMatrix,
        judgments: list[CoverageGapJudgment],
        *,
        library: EvidenceLibrary,
    ) -> CoverageMatrix:
        """Let the evidence judge settle every cell retrieval only reached.

        Retrieval produces candidate cells, never verified ones, so a judgment may
        promote a candidate to covered or reject it as missing or not_stated. A
        covered verdict is only honoured when the cited Evidence IDs exist in the
        library and belong to the paper being judged; otherwise the claim is
        downgraded to a retrieval gap, because a cell must never look answered
        merely because the model asserted it.
        """

        evidence_paper_ids = {item.evidence_id: item.paper_id for item in library.evidence}
        for judgment in judgments:
            existing_cell = matrix.cell_for(judgment.paper_id, judgment.dimension)
            candidate_ids = (
                existing_cell.evidence_ids if existing_cell is not None else []
            )
            status = CoverageStatus(judgment.status)
            note = judgment.reason
            verified_ids: list[str] = []
            if status is CoverageStatus.COVERED:
                verified_ids = [
                    evidence_id
                    for evidence_id in judgment.evidence_ids
                    if evidence_paper_ids.get(evidence_id) == judgment.paper_id
                ]
                if not verified_ids:
                    status = CoverageStatus.MISSING
                    note = (
                        "Judgment claimed coverage without evidence from this paper: "
                        f"{judgment.reason}"
                    )[:400]
            matrix = matrix.with_cell(
                CoverageCell(
                    paper_id=judgment.paper_id,
                    dimension=judgment.dimension,
                    status=status,
                    evidence_ids=candidate_ids,
                    verified_evidence_ids=verified_ids,
                    note=note,
                )
            )
        return matrix

    @staticmethod
    def _evidence_library(state: ReaderState) -> EvidenceLibrary:
        evidence: list[Evidence] = []
        seen_chunks: set[str] = set()
        for report in state["retrieval_reports"]:
            for hit in report.hits:
                if hit.node_id in seen_chunks:
                    continue
                seen_chunks.add(hit.node_id)
                evidence.append(
                    Evidence(
                        evidence_id=ReaderAgentGraph._evidence_id(hit.node_id),
                        paper_id=hit.paper_id,
                        paper_title=hit.paper_title or hit.paper_id,
                        chunk_id=hit.node_id,
                        section_path=hit.section_path,
                        page_start=hit.page_start,
                        page_end=hit.page_end,
                        raw_text=hit.text,
                        evidence_text=hit.text,
                        retrieval_score=hit.ranking_score,
                        rerank_score=hit.rerank_score,
                        spans=hit.spans,
                    )
                )
        document = state["pdf_document"]
        candidate = state["pdf_candidate"]
        if not evidence and document is not None:
            paper_id = candidate.paper.paper_id if candidate is not None else "external-paper"
            title = candidate.paper.title if candidate is not None else paper_id
            evidence.append(
                Evidence(
                    evidence_id=ReaderAgentGraph._evidence_id(
                        f"{paper_id}:pdf-extraction"
                    ),
                    paper_id=paper_id,
                    paper_title=title,
                    chunk_id=f"{paper_id}:pdf-extraction",
                    section_path=[],
                    page_start=1,
                    page_end=document.extracted_pages or None,
                    raw_text=document.text,
                    evidence_text=document.text,
                    retrieval_score=1.0,
                )
            )
        if not evidence and state["local_metadata"] is not None:
            metadata_text = json.dumps(state["local_metadata"], ensure_ascii=False)
            paper_id = (
                state["task"].target_paper_ids[0]
                if state["task"].target_paper_ids
                else "local-metadata"
            )
            title_value = state["local_metadata"].get("title")
            if not isinstance(title_value, str):
                title_value = next(
                    (
                        item.get("title")
                        for item in state["local_metadata"].values()
                        if isinstance(item, dict)
                    ),
                    None,
                )
            title = title_value if isinstance(title_value, str) else paper_id
            evidence.append(
                Evidence(
                    evidence_id=ReaderAgentGraph._evidence_id(f"{paper_id}:metadata"),
                    paper_id=paper_id,
                    paper_title=title,
                    chunk_id=f"{paper_id}:metadata",
                    section_path=[],
                    raw_text=metadata_text,
                    evidence_text=metadata_text,
                    retrieval_score=1.0,
                )
            )
        return EvidenceLibrary(objective=state["task"].objective, evidence=evidence)

    @staticmethod
    def _evidence_id(chunk_id: str) -> str:
        return f"E-{hashlib.sha256(chunk_id.encode('utf-8')).hexdigest()[:12]}"

    @staticmethod
    def _validate_evidence_references(
        report: ReadingReport,
        library: EvidenceLibrary,
    ) -> ReadingReport:
        valid_ids = {item.evidence_id for item in library.evidence}
        valid_evidence = [item for item in report.evidence if item.evidence_id in valid_ids]
        invalid_count = len(report.evidence) - len(valid_evidence)
        if invalid_count == 0:
            return report
        return report.model_copy(
            update={
                "evidence": valid_evidence,
                "limitations": [
                    *report.limitations,
                    f"Removed {invalid_count} evidence reference(s) not present in the library",
                ][:8],
            }
        )

    async def _fetch_pdf(
        self,
        context: AgentRunContext,
        candidate: PdfCandidate,
    ) -> tuple[PdfDocument | None, str | None]:
        if candidate.paper.pdf_url is None or self._document_gateway is None:
            return None, "Selected paper has no downloadable PDF URL"
        call_id = f"pdf-{uuid4()}"
        arguments: dict[str, JsonValue] = {
            "paper_id": candidate.paper.paper_id,
            "url": str(candidate.paper.pdf_url),
        }
        await self._publish_tool_started(context, call_id, "fetch_arxiv_pdf", arguments)
        started = perf_counter()
        document: PdfDocument | None = None
        error_message: str | None = None
        summary: dict[str, JsonValue] | None = None
        try:
            document = await self._document_gateway.fetch(str(candidate.paper.pdf_url))
            summary = {
                "paper_id": candidate.paper.paper_id,
                "page_count": document.page_count,
                "extracted_pages": document.extracted_pages,
                "extracted_characters": document.extracted_characters,
                "truncated": document.truncated,
                "warnings": cast(JsonValue, document.extraction_warnings),
            }
            await context.publisher.publish(
                EventType.TOOL_COMPLETED.value,
                {
                    "source": DecisionSource.EXTERNAL.value,
                    "actor": "fetch_arxiv_pdf",
                    "tool_call_id": call_id,
                    "tool_name": "fetch_arxiv_pdf",
                    **summary,
                    "duration_ms": int((perf_counter() - started) * 1000),
                },
            )
        except (ValueError, RuntimeError) as error:
            error_message = str(error)
            await self._publish_tool_failure(
                context, call_id, "fetch_arxiv_pdf", error_message, started
            )
        await self._record_tool(
            context=context,
            call_id=call_id,
            tool_name="fetch_arxiv_pdf",
            arguments=arguments,
            summary=summary or {"error": error_message},
            error=error_message,
            started=started,
        )
        return document, error_message

    async def _retrieve_indexed_paper(
        self,
        context: AgentRunContext,
        *,
        paper_ids: list[str],
        queries: list[ReadingSubQuestion],
        strategy: RetrievalStrategy | None,
        stage: str,
    ) -> tuple[TreeRetrievalReport | None, str | None]:
        if self._paper_retriever is None:
            return None, "TreeRAG retriever is not configured"
        call_id = f"rag-{uuid4()}"
        plan_queries = [
            RetrievalQuery(
                query=item.retrieval_query,
                mode=item.mode,
                paper_ids=item.paper_ids or paper_ids,
                dimension=item.dimension,
            )
            for item in queries
        ]
        primary = plan_queries[0]
        # The plan scopes each sub-question to the paper it reads, so the overall
        # scope is the union: reporting only the first sub-question's paper would
        # understate the comparison and shrink its budget.
        scoped_paper_ids = list(
            dict.fromkeys(
                [
                    *paper_ids,
                    *(paper_id for item in plan_queries for paper_id in item.paper_ids),
                ]
            )
        )
        arguments: dict[str, JsonValue] = {
            "paper_ids": cast(JsonValue, scoped_paper_ids),
            "retrieval_strategy": strategy.value if strategy is not None else None,
            "sub_questions": cast(
                JsonValue,
                [item.model_dump(mode="json") for item in plan_queries],
            ),
        }
        await self._publish_tool_started(context, call_id, "retrieve_indexed_paper", arguments)
        started = perf_counter()
        report: TreeRetrievalReport | None = None
        error_message: str | None = None
        try:
            report = await self._paper_retriever.retrieve(
                primary.query,
                paper_ids=scoped_paper_ids or None,
                mode=primary.mode,
                strategy=strategy,
                queries=plan_queries,
            )
            if not report.hits:
                raise ValueError("No evidence chunks were retrieved from the requested scope")
            await context.publisher.publish(
                EventType.TOOL_COMPLETED.value,
                {
                    "source": DecisionSource.EXTERNAL.value,
                    "actor": "retrieve_indexed_paper",
                    "stage": stage,
                    "tool_call_id": call_id,
                    "tool_name": "retrieve_indexed_paper",
                    "retrieval_strategy": report.strategy.value,
                    "query_count": len(report.queries),
                    "missing_paper_ids": cast(JsonValue, report.missing_paper_ids),
                    "candidate_ratio": (
                        report.coverage.candidate_ratio
                        if report.coverage is not None
                        else None
                    ),
                    "coverage": (
                        cast(JsonValue, report.coverage.model_dump(mode="json"))
                        if report.coverage is not None
                        else None
                    ),
                    "keyword_candidate_count": report.keyword_candidate_count,
                    "keyword_pool_size": report.keyword_pool_size,
                    "keyword_truncated_terms": cast(
                        JsonValue, report.keyword_truncated_terms
                    ),
                    "initial_hit_count": report.initial_hit_count,
                    "expanded_candidate_count": report.expanded_candidate_count,
                    "deduplicated_candidate_count": report.deduplicated_candidate_count,
                    "mmr_candidate_count": report.mmr_candidate_count,
                    "reranker_name": report.reranker_name,
                    "reranker_applied": report.reranker_applied,
                    "reranker_error": report.reranker_error,
                    "hit_count": len(report.hits),
                    "searched_globally": report.searched_globally,
                    "candidate_paper_ids": cast(JsonValue, report.candidate_paper_ids),
                    "hits": cast(
                        JsonValue,
                        [
                            {
                                "rank": hit.rank,
                                "paper_id": hit.paper_id,
                                "paper_title": hit.paper_title,
                                "section_path": hit.section_path,
                                "semantic_role": hit.semantic_role,
                                "block_types": [item.value for item in hit.block_types],
                                "object_labels": hit.object_labels,
                                "page_start": hit.page_start,
                                "page_end": hit.page_end,
                                "text": hit.text,
                                "matched_dimensions": hit.matched_dimensions,
                                "matched_queries": hit.matched_queries,
                                "source": hit.source.value,
                                "figure_asset": hit.figure_asset,
                                "figure_caption": hit.figure_caption,
                                "raw_asset_ref": hit.raw_asset_ref,
                                "table_rows": hit.table_rows,
                            }
                            for hit in report.hits
                        ],
                    ),
                    "duration_ms": int((perf_counter() - started) * 1000),
                },
            )
        except (ValueError, RuntimeError) as error:
            error_message = str(error)
            await self._publish_tool_failure(
                context, call_id, "retrieve_indexed_paper", error_message, started
            )
        summary: dict[str, JsonValue] = (
            {
                "retrieval_strategy": report.strategy.value,
                "query_count": len(report.queries),
                "missing_paper_ids": cast(JsonValue, report.missing_paper_ids),
                "candidate_ratio": (
                    report.coverage.candidate_ratio if report.coverage is not None else None
                ),
                "keyword_candidate_count": report.keyword_candidate_count,
                "keyword_pool_size": report.keyword_pool_size,
                "keyword_truncated_terms": cast(
                    JsonValue, report.keyword_truncated_terms
                ),
                "initial_hit_count": report.initial_hit_count,
                "expanded_candidate_count": report.expanded_candidate_count,
                "deduplicated_candidate_count": report.deduplicated_candidate_count,
                "mmr_candidate_count": report.mmr_candidate_count,
                "reranker_name": report.reranker_name,
                "reranker_applied": report.reranker_applied,
                "reranker_error": report.reranker_error,
                "hit_count": len(report.hits),
                "searched_globally": report.searched_globally,
                "candidate_paper_ids": cast(JsonValue, report.candidate_paper_ids),
            }
            if report is not None
            else {"error": error_message}
        )
        await self._record_tool(
            context=context,
            call_id=call_id,
            tool_name="retrieve_indexed_paper",
            arguments=arguments,
            summary=summary,
            error=error_message,
            started=started,
        )
        return report, error_message

    @staticmethod
    async def _publish_tool_started(
        context: AgentRunContext,
        call_id: str,
        tool_name: str,
        arguments: dict[str, JsonValue],
    ) -> None:
        await context.publisher.publish(
            EventType.TOOL_STARTED.value,
            {
                "source": DecisionSource.TOOL.value,
                "actor": tool_name,
                "requested_by": "reader",
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )

    @staticmethod
    async def _publish_tool_failure(
        context: AgentRunContext,
        call_id: str,
        tool_name: str,
        error: str,
        started: float,
    ) -> None:
        await context.publisher.publish(
            EventType.TOOL_FAILED.value,
            {
                "source": DecisionSource.EXTERNAL.value,
                "actor": tool_name,
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "error": error,
                "duration_ms": int((perf_counter() - started) * 1000),
            },
        )

    async def _record_tool(
        self,
        *,
        context: AgentRunContext,
        call_id: str,
        tool_name: str,
        arguments: dict[str, JsonValue],
        summary: dict[str, JsonValue],
        error: str | None,
        started: float,
    ) -> None:
        if self._recorder is None:
            return
        await self._recorder.record_tool_execution(
            context=context,
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            content=json.dumps(summary, ensure_ascii=False),
            result_summary=summary,
            error=error,
            duration_ms=int((perf_counter() - started) * 1000),
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
        decision = state["decision"]
        if decision is None or not isinstance(decision.task, ReaderTask):
            raise ValueError("Reader Agent requires a reader decision")
        task = decision.task
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
                selected_artifacts=selected,
                is_local_paper=is_local_paper,
                local_metadata=None,
                pdf_candidate=(
                    self._select_pdf_candidate(selected, task.paper_id)
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
        attempted_queries = result["attempted_queries"]
        unique_hit_count = self._unique_hit_count(retrieval_reports)
        coverage = ReaderAgentGraph._coverage_matrix(result)
        missing_paper_ids = list(
            dict.fromkeys(
                paper_id
                for report_item in retrieval_reports
                for paper_id in report_item.missing_paper_ids
            )
        )
        strategy = next(
            (
                report_item.strategy
                for report_item in reversed(retrieval_reports)
                if report_item.strategy is not None
            ),
            None,
        )
        supervisor_summary = self._compact_summary(report.analysis_summary)
        evidence_library = ReaderAgentGraph._evidence_library(result)
        artifact = AgentArtifact(
            title=f"Reading report: {report.paper_or_material}",
            supervisor_summary=ReaderAgentSummary(
                summary=supervisor_summary,
                paper_or_material=report.paper_or_material,
                objective_satisfied=report.objective_satisfied,
                answered_points=report.answered_points,
                blocking_gaps=report.blocking_gaps,
                evidence_scope=report.evidence_scope,
                limitations=report.limitations,
                pdf_truncated=pdf_document.truncated if pdf_document else False,
                pdf_error=result["material_error"],
                retrieval_mode=(retrieval_reports[-1].mode if retrieval_reports else None),
                retrieval_strategy=strategy,
                retrieval_hit_count=unique_hit_count,
                retrieval_rounds=result["retrieval_attempts"],
                attempted_queries=attempted_queries,
                target_paper_ids=targets,
                missing_paper_ids=missing_paper_ids,
                coverage=coverage,
            ),
            content=json.dumps(
                {
                    "reading_report": report.model_dump(mode="json"),
                    "evidence_library": evidence_library.model_dump(mode="json"),
                    "coverage_matrix": (
                        coverage.model_dump(mode="json") if coverage is not None else None
                    ),
                },
                ensure_ascii=False,
            ),
            source_artifact_ids=[item.id for item in selected],
        )
        judgment = result["assessment"]
        next_decision: SupervisorDecision | None = None
        if task.depth is ReaderDepth.QUICK:
            missing = judgment.missing_requirements if judgment is not None else []
            next_decision = SupervisorDecision(
                assessment=DecisionAssessment(
                    observations=["Quick Reader completed its single scoped evidence pass"],
                    missing_information=missing,
                    decision_summary=(
                        "The quick path is complete; answer the exact question from its evidence"
                    ),
                ),
                task=WriterTask(
                    objective=(
                        "Answer the user's exact question directly and concisely using the Reader "
                        "artifact; state only material evidence limitations"
                    ),
                    source_artifact_ids=[artifact.id],
                ),
            )
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "reader",
                "stage": "reader",
                "summary": supervisor_summary,
                "artifact_id": str(artifact.id),
                "source_count": len(selected),
                "evidence_count": len(report.evidence),
                "objective_satisfied": report.objective_satisfied,
                "answered_points": cast(JsonValue, report.answered_points),
                "blocking_gaps": cast(JsonValue, report.blocking_gaps),
                "evidence_scope": report.evidence_scope,
                "limitations": cast(JsonValue, report.limitations),
                "pdf_pages": pdf_document.page_count if pdf_document else 0,
                "pdf_truncated": pdf_document.truncated if pdf_document else False,
                "retrieval_rounds": result["retrieval_attempts"],
                "attempted_queries": cast(JsonValue, attempted_queries),
                "retrieval_hit_count": unique_hit_count,
                "retrieval_strategy": strategy.value if strategy is not None else None,
                "target_paper_ids": cast(JsonValue, targets),
                "missing_paper_ids": cast(JsonValue, missing_paper_ids),
                "coverage": (
                    cast(JsonValue, coverage.model_dump(mode="json"))
                    if coverage is not None
                    else None
                ),
                "coverage_ratio": (
                    coverage.coverage_ratio if coverage is not None else None
                ),
                "candidate_ratio": (
                    coverage.candidate_ratio if coverage is not None else None
                ),
                "candidate_cell_count": (
                    coverage.candidate_cell_count if coverage is not None else None
                ),
                "verified_cell_count": (
                    coverage.covered_cell_count if coverage is not None else None
                ),
                "unresolved_cells": cast(
                    JsonValue,
                    (
                        [
                            cell.model_dump(mode="json")
                            for cell in coverage.unresolved_cells()
                        ]
                        if coverage is not None
                        else []
                    ),
                ),
            },
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
            "decision": next_decision,
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
    def _compact_summary(value: str, max_length: int = 1_000) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= max_length:
            return normalized
        return f"{normalized[: max_length - 3].rstrip()}..."

    @staticmethod
    def _unique_hit_count(reports: list[TreeRetrievalReport]) -> int:
        return len({hit.node_id for report in reports for hit in report.hits})

    @staticmethod
    def _select_pdf_candidate(
        artifacts: list[AgentArtifact],
        requested_paper_id: str | None,
    ) -> PdfCandidate | None:
        candidates: list[PdfCandidate] = []
        for artifact in artifacts:
            if artifact.kind is not ArtifactKind.SEARCH_RESULT:
                continue
            try:
                report = SearchReport.model_validate_json(artifact.content)
                papers = report.papers
            except ValueError:
                try:
                    raw = json.loads(artifact.content)
                    papers = (
                        [Paper.model_validate(item) for item in raw]
                        if isinstance(raw, list)
                        else []
                    )
                except (json.JSONDecodeError, ValueError):
                    papers = []
            candidates.extend(
                PdfCandidate(paper=paper, source_artifact_id=str(artifact.id)) for paper in papers
            )
        return next(
            (item for item in candidates if item.paper.paper_id == requested_paper_id),
            None,
        )
