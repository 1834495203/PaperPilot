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
from app.domain.rag import RetrievalMode, TreeRetrievalReport
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway, ModelUsage
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    AgentName,
    ArtifactKind,
    CompletedStep,
    DecisionAssessment,
    DecisionSource,
    ReaderAgentSummary,
    ReaderDepth,
    ReaderTask,
    ReadingEvidenceAssessment,
    ReadingPlan,
    ReadingReport,
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
    selected_artifacts: list[AgentArtifact]
    is_local_paper: bool
    local_metadata: dict[str, JsonValue] | None
    pdf_candidate: PdfCandidate | None
    pdf_document: PdfDocument | None
    material_error: str | None
    plan: ReadingPlan | None
    assessment: ReadingEvidenceAssessment | None
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
    retrieval_reports: list[TreeRetrievalReport]
    retrieval_attempts: int
    attempted_queries: list[str]
    final_report: ReadingReport | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int


PreparationRoute = Literal["quick_plan", "plan"]
ReaderRoute = Literal["retrieve", "fetch", "judge"]
EvidenceRoute = Literal["retrieve", "synthesize"]


class ReaderAgentGraph:
    """Plan-retrieve-assess-read LangGraph subgraph for local or external evidence."""

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
        builder.add_node("quick_plan", self._quick_plan)
        builder.add_node("plan", self._plan)
        builder.add_node("fetch", self._fetch)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("judge", self._judge)
        builder.add_node("synthesize", self._synthesize)
        builder.add_edge(START, "prepare")
        builder.add_conditional_edges(
            "prepare",
            self._route_after_prepare,
            {"quick_plan": "quick_plan", "plan": "plan"},
        )
        builder.add_conditional_edges(
            "quick_plan",
            self._route_after_plan,
            {"retrieve": "retrieve", "fetch": "fetch", "judge": "judge"},
        )
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"retrieve": "retrieve", "fetch": "fetch", "judge": "judge"},
        )
        builder.add_edge("fetch", "judge")
        builder.add_edge("retrieve", "judge")
        builder.add_conditional_edges(
            "judge",
            self._route_after_judgment,
            {"retrieve": "retrieve", "synthesize": "synthesize"},
        )
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
        if state["is_local_paper"]:
            if self._paper_library is not None and task.paper_id is not None:
                try:
                    paper = await self._paper_library.get_paper(task.paper_id)
                    local_metadata = cast(
                        dict[str, JsonValue],
                        paper.metadata.model_dump(mode="json"),
                    )
                except PaperNotFoundError:
                    material_error = "Local paper metadata was not found"
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

    async def _quick_plan(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        task = state["task"]
        needs_retrieval = (
            state["is_local_paper"] and self._paper_retriever is not None
        ) or (not state["is_local_paper"] and self._document_gateway is not None)
        plan = ReadingPlan(
            needs_retrieval=needs_retrieval,
            query=task.objective if needs_retrieval else None,
            mode=RetrievalMode.METHOD if needs_retrieval else None,
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
            },
        )
        return {"plan": plan}

    async def _plan(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        context = runtime.context
        task = state["task"]
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "reader",
                "stage": "reader.plan",
                "summary": "Reader 正在制定论文库检索计划",
                "paper_ids": [] if task.paper_id is None else [task.paper_id],
                "retrieval_scope": (
                    "all_local_papers" if task.paper_id is None else "single_paper"
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
                            "paper_id": task.paper_id,
                            "retrieval_scope": (
                                "all_local_papers"
                                if task.paper_id is None
                                else "single_paper"
                            ),
                            "paper_metadata": state["local_metadata"],
                            "reading_objective": task.objective,
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            ReadingPlan,
        )
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "reader",
                "stage": "reader.plan",
                "summary": result.value.rationale,
                "needs_retrieval": result.value.needs_retrieval,
                "query": result.value.query,
                "retrieval_mode": (
                    result.value.mode.value if result.value.mode is not None else None
                ),
                "evidence_requirements": cast(JsonValue, result.value.evidence_requirements),
            },
        )
        return {"plan": result.value, **self._usage_update(state, result.usage)}

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
        assessment = state["assessment"]
        if plan is None:
            raise ValueError("Reader retrieval requires a reading plan")
        query = assessment.next_query if assessment is not None else plan.query
        mode = assessment.next_mode if assessment is not None else plan.mode
        if query is None or mode is None:
            raise ValueError("Reader retry requires a query and retrieval mode")
        report, error = await self._retrieve_indexed_paper(
            runtime.context,
            paper_id=state["task"].paper_id,
            query=query,
            mode=mode,
        )
        reports = list(state["retrieval_reports"])
        if report is not None:
            reports.append(report)
        return {
            "retrieval_reports": reports,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
            "attempted_queries": [*state["attempted_queries"], query],
            "material_error": error,
            "assessment": None,
            "tool_calls": state["tool_calls"] + 1,
        }

    async def _judge(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        context = runtime.context
        plan = state["plan"]
        if plan is None:
            raise ValueError("Reader evidence assessment requires a reading plan")
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "reader",
                "stage": "reader.judge",
                "summary": "Judge 正在按用户原问题检查证据充分性",
                "retrieval_round": len(state["retrieval_reports"]),
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
                            "reading_plan": plan.model_dump(mode="json"),
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
                "stage": "reader.judge",
                "summary": assessment.coverage_summary,
                "evidence_sufficient": assessment.evidence_sufficient,
                "covered_requirements": cast(JsonValue, assessment.covered_requirements),
                "missing_requirements": cast(JsonValue, assessment.missing_requirements),
                "retry_recommended": assessment.retry_recommended,
                "next_query": assessment.next_query,
                "next_mode": (
                    assessment.next_mode.value if assessment.next_mode is not None else None
                ),
            },
        )
        return {
            "assessment": assessment,
            **self._usage_update(state, result.usage),
        }

    async def _synthesize(
        self,
        state: ReaderState,
        runtime: Runtime[AgentRunContext],
    ) -> ReaderStateUpdate:
        del runtime
        task = state["task"]
        prompt = (
            f"Assigned evidence-reading task:\n{task.objective}\n\n"
            f"Reader depth:\n{task.depth.value}\n\n"
            f"Retrieval scope:\n"
            f"{'all locally indexed papers' if task.paper_id is None else task.paper_id}\n\n"
            f"Available material:\n{self._reading_material(state)}\n\n"
            "Analyze only the supplied evidence. Do not evaluate the overall workflow."
        )
        result = await self._model.generate_structured(
            [SystemMessage(content=READER_PROMPT), HumanMessage(content=prompt)],
            ReadingReport,
        )
        return {
            "final_report": result.value,
            **self._usage_update(state, result.usage),
        }

    @staticmethod
    def _route_after_prepare(state: ReaderState) -> PreparationRoute:
        return "quick_plan" if state["task"].depth is ReaderDepth.QUICK else "plan"

    def _route_after_plan(self, state: ReaderState) -> ReaderRoute:
        plan = state["plan"]
        if plan is None:
            raise ValueError("Reader planning node did not produce a plan")
        if not plan.needs_retrieval:
            return "judge"
        if state["is_local_paper"] and self._paper_retriever is not None:
            return "retrieve"
        if not state["is_local_paper"] and self._document_gateway is not None:
            return "fetch"
        return "judge"

    def _route_after_judgment(self, state: ReaderState) -> EvidenceRoute:
        assessment = state["assessment"]
        if assessment is None:
            raise ValueError("Reader assessment node did not produce a decision")
        if (
            state["task"].depth is ReaderDepth.DEEP
            and state["is_local_paper"]
            and self._paper_retriever is not None
            and not assessment.evidence_sufficient
            and assessment.retry_recommended
            and state["retrieval_attempts"] < self._max_retrieval_rounds
        ):
            return "retrieve"
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
    def _reading_material(state: ReaderState) -> str:
        if state["retrieval_reports"]:
            return json.dumps(
                {
                    "paper_id": state["task"].paper_id,
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
                        report.model_dump(mode="json") for report in state["retrieval_reports"]
                    ],
                    "instruction": (
                        "Treat paper_metadata as authoritative bibliographic metadata. "
                        "Use only retrieved chunks as evidence for substantive claims and "
                        "preserve page numbers and section paths in evidence entries."
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
                    "full_text": document.text,
                },
                ensure_ascii=False,
            )
        material = json.dumps(
            {
                "paper_metadata": state["local_metadata"],
                "paper": candidate.paper.model_dump(mode="json") if candidate else None,
            },
            ensure_ascii=False,
        )
        return (
            f"{material}\n\nMaterial retrieval failed: {state['material_error']}"
            if state["material_error"]
            else material
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
        paper_id: str | None,
        query: str,
        mode: RetrievalMode,
    ) -> tuple[TreeRetrievalReport | None, str | None]:
        if self._paper_retriever is None:
            return None, "TreeRAG retriever is not configured"
        call_id = f"rag-{uuid4()}"
        arguments: dict[str, JsonValue] = {
            "paper_id": paper_id,
            "retrieval_scope": "all_local_papers" if paper_id is None else "single_paper",
            "query": query,
            "mode": mode.value,
        }
        await self._publish_tool_started(context, call_id, "retrieve_indexed_paper", arguments)
        started = perf_counter()
        report: TreeRetrievalReport | None = None
        error_message: str | None = None
        try:
            report = await self._paper_retriever.retrieve(
                query,
                paper_ids=None if paper_id is None else [paper_id],
                mode=mode,
            )
            if not report.hits:
                raise ValueError("No evidence chunks were retrieved from the requested scope")
            await context.publisher.publish(
                EventType.TOOL_COMPLETED.value,
                {
                    "source": DecisionSource.EXTERNAL.value,
                    "actor": "retrieve_indexed_paper",
                    "tool_call_id": call_id,
                    "tool_name": "retrieve_indexed_paper",
                    "initial_hit_count": report.initial_hit_count,
                    "expanded_candidate_count": report.expanded_candidate_count,
                    "hit_count": len(report.hits),
                    "searched_globally": report.searched_globally,
                    "candidate_paper_ids": cast(JsonValue, report.candidate_paper_ids),
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
                "initial_hit_count": report.initial_hit_count,
                "expanded_candidate_count": report.expanded_candidate_count,
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
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "reader",
                "stage": "reader",
                "summary": "Reader Agent 子图开始执行",
                "objective": task.objective,
                "reader_depth": task.depth.value,
                "paper_ids": [] if task.paper_id is None else [task.paper_id],
                "retrieval_scope": (
                    "all_local_papers" if task.paper_id is None else "single_paper"
                ),
            },
        )
        selected = select_artifacts(
            state["artifacts"],
            [task.source_artifact_id] if task.source_artifact_id is not None else [],
        )
        is_local_paper = (
            task.paper_id in context.paper_ids
            if task.paper_id is not None
            else context.local_corpus_available or bool(context.paper_ids)
        )
        result = await self._graph.run(
            ReaderState(
                task=task,
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
        supervisor_summary = self._compact_summary(report.analysis_summary)
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
                retrieval_hit_count=unique_hit_count,
                retrieval_rounds=result["retrieval_attempts"],
                attempted_queries=attempted_queries,
            ),
            content=report.model_dump_json(),
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
        requested_paper_id: str,
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
