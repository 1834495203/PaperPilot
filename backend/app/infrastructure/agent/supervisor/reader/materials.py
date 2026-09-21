"""Material acquisition for the Reader: metadata, PDFs and indexed retrieval.

This module owns every call to the underlying library, gateway and retriever, and
the tool events and execution records that describe them. It returns material and
errors only; judging whether that material is sufficient, and whether another
retrieval round is worth it, stays with the graph nodes and the model.
"""

import json
from dataclasses import dataclass
from time import perf_counter
from typing import cast
from uuid import uuid4

from app.application.agent import AgentRunContext
from app.application.paper_library import PaperLibraryService, PaperNotFoundError
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.enums import EventType
from app.domain.papers import Paper, PdfDocument
from app.domain.ports import PaperDocumentGateway
from app.domain.rag import (
    RetrievalQuery,
    RetrievalStrategy,
    TreeRetrievalReport,
)
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    ArtifactKind,
    DecisionSource,
    ReaderTask,
    ReadingSubQuestion,
    SearchReport,
)
from app.infrastructure.agent.supervisor.reader.state import PdfCandidate


def select_pdf_candidate(
    artifacts: list[AgentArtifact],
    requested_paper_id: str | None,
) -> PdfCandidate | None:
    """Find the requested paper inside the selected search artifacts."""

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


@dataclass(frozen=True, slots=True)
class PdfFetch:
    """Outcome of one PDF fetch.

    ``attempted`` is false when the candidate itself was unusable, so a caller can
    tell a failed tool call from a fetch that never happened.
    """

    document: PdfDocument | None
    error: str | None
    attempted: bool


class ReaderMaterials:
    """Fetches what Reader reads, and records how each fetch went."""

    def __init__(
        self,
        *,
        document_gateway: PaperDocumentGateway | None = None,
        paper_retriever: TreeRagRetriever | None = None,
        paper_library: PaperLibraryService | None = None,
        recorder: AgentExecutionRecorder | None = None,
    ) -> None:
        self._document_gateway = document_gateway
        self._paper_retriever = paper_retriever
        self._paper_library = paper_library
        self._recorder = recorder

    @property
    def can_retrieve_indexed_papers(self) -> bool:
        return self._paper_retriever is not None

    @property
    def can_fetch_pdf(self) -> bool:
        return self._document_gateway is not None

    async def prepare_metadata(
        self,
        task: ReaderTask,
        *,
        is_local_paper: bool,
        pdf_candidate: PdfCandidate | None,
    ) -> tuple[dict[str, JsonValue] | None, str | None]:
        """Resolve the authoritative metadata for the requested reading scope."""

        local_metadata: dict[str, JsonValue] | None = None
        material_error: str | None = None
        targets = task.target_paper_ids
        if is_local_paper:
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
        elif pdf_candidate is None:
            material_error = (
                "The requested paper ID was not found in the selected search artifact"
            )
        else:
            local_metadata = cast(
                dict[str, JsonValue], pdf_candidate.paper.model_dump(mode="json")
            )
        return local_metadata, material_error

    async def fetch_pdf(
        self,
        context: AgentRunContext,
        candidate: PdfCandidate | None,
    ) -> PdfFetch:
        if candidate is None:
            return PdfFetch(None, "No external PDF candidate is available", attempted=False)
        if self._document_gateway is None:
            return PdfFetch(
                None, "PDF document gateway is not configured", attempted=False
            )
        if candidate.paper.pdf_url is None:
            return PdfFetch(
                None, "Selected paper has no downloadable PDF URL", attempted=False
            )
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
        return PdfFetch(document, error_message, attempted=True)

    async def retrieve(
        self,
        context: AgentRunContext,
        *,
        paper_ids: list[str],
        queries: list[ReadingSubQuestion],
        strategy: RetrievalStrategy | None,
        stage: str,
    ) -> tuple[TreeRetrievalReport | None, str | None]:
        """Run one indexed retrieval round over the planned sub-questions."""

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
