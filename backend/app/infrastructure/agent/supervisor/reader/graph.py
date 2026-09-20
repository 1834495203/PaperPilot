"""The Reader subgraph: plan, retrieve, judge, repair, synthesize."""

from typing import cast

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.application.paper_library import PaperLibraryService
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.enums import EventType
from app.domain.ports import PaperDocumentGateway
from app.domain.rag import (
    CoverageMatrix,
    EvidenceLibrary,
    RetrievalMode,
    RetrievalStrategy,
)
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway, ModelUsage
from app.infrastructure.agent.supervisor.models import (
    DecisionSource,
    ReaderDepth,
    ReadingEvidenceAssessment,
    ReadingPlan,
    ReadingReport,
    ReadingSubQuestion,
)
from app.infrastructure.agent.supervisor.reader.evidence import (
    build_evidence_library,
    merge_coverage,
    merge_judgments,
    validate_report_references,
)
from app.infrastructure.agent.supervisor.reader.materials import ReaderMaterials
from app.infrastructure.agent.supervisor.reader.prompts import (
    build_assessment_messages,
    build_planning_messages,
    build_synthesis_messages,
)
from app.infrastructure.agent.supervisor.reader.state import (
    EvidenceRoute,
    PlanRoute,
    ReaderState,
    ReaderStateUpdate,
)


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
        self._materials = ReaderMaterials(
            document_gateway=document_gateway,
            paper_retriever=paper_retriever,
            paper_library=paper_library,
            recorder=recorder,
        )
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
        local_metadata, material_error = await self._materials.prepare_metadata(
            state["task"],
            is_local_paper=state["is_local_paper"],
            pdf_candidate=state["pdf_candidate"],
        )
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
            build_planning_messages(state),
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
            state["is_local_paper"] and self._materials.can_retrieve_indexed_papers
        ) or (not state["is_local_paper"] and self._materials.can_fetch_pdf)
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
        fetch = await self._materials.fetch_pdf(runtime.context, state["pdf_candidate"])
        update: ReaderStateUpdate = {
            "pdf_document": fetch.document,
            "material_error": fetch.error,
        }
        if fetch.attempted:
            update["tool_calls"] = state["tool_calls"] + 1
        return update

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
        """Execute one retrieval round within the configured round budget.

        Both routes into a round already require a remaining attempt, so reaching
        this point at capacity means the budget was bypassed: refuse instead of
        quietly running a trimmed round that the policy never allowed.
        """

        if state["retrieval_attempts"] >= self._max_retrieval_rounds:
            raise ValueError(
                "Reader retrieval budget of "
                f"{self._max_retrieval_rounds} round(s) is already spent"
            )
        report, error = await self._materials.retrieve(
            context,
            paper_ids=state["task"].target_paper_ids,
            queries=sub_questions,
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
                *(item.retrieval_query for item in sub_questions),
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
        library = self._evidence_library(state)
        coverage = self._coverage_matrix(state, library)
        can_retry = (
            state["task"].depth is ReaderDepth.DEEP
            and state["is_local_paper"]
            and self._materials.can_retrieve_indexed_papers
            and state["retrieval_attempts"] < self._max_retrieval_rounds
        )
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
            build_assessment_messages(
                state,
                library,
                coverage,
                can_retry=can_retry,
            ),
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
            "coverage_judgments": merge_judgments(
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
        library = self._evidence_library(state)
        coverage = self._coverage_matrix(state, library)
        result = await self._model.generate_structured(
            build_synthesis_messages(state, library, coverage),
            ReadingReport,
        )
        report = validate_report_references(result.value, library)
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
        if state["is_local_paper"] and self._materials.can_retrieve_indexed_papers:
            return "retrieve"
        if not state["is_local_paper"] and self._materials.can_fetch_pdf:
            return "fetch"
        return "assess"

    def _route_after_assessment(self, state: ReaderState) -> EvidenceRoute:
        assessment = state["assessment"]
        if assessment is None:
            raise ValueError("Reader assessment node did not produce a decision")
        if (
            state["task"].depth is ReaderDepth.DEEP
            and state["is_local_paper"]
            and self._materials.can_retrieve_indexed_papers
            and not assessment.evidence_sufficient
            and assessment.retry_recommended
            and bool(assessment.gap_queries())
            and state["retrieval_attempts"] < self._max_retrieval_rounds
        ):
            return "retrieve_gaps"
        return "synthesize"

    @staticmethod
    def _usage_update(state: ReaderState, usage: ModelUsage) -> ReaderStateUpdate:
        return {
            "input_tokens": state["input_tokens"] + usage.input_tokens,
            "output_tokens": state["output_tokens"] + usage.output_tokens,
            "total_tokens": state["total_tokens"] + usage.total_tokens,
            "llm_calls": state["llm_calls"] + 1,
        }

    @staticmethod
    def _evidence_library(state: ReaderState) -> EvidenceLibrary:
        return build_evidence_library(
            state["task"],
            state["retrieval_reports"],
            state["pdf_document"],
            state["pdf_candidate"],
            state["local_metadata"],
        )

    def _coverage_matrix(
        self,
        state: ReaderState,
        library: EvidenceLibrary,
    ) -> CoverageMatrix | None:
        return merge_coverage(
            state["retrieval_reports"],
            state["plan"],
            state["coverage_judgments"],
            library,
        )
