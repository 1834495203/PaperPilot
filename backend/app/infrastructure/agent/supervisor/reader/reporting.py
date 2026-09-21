"""Reader artifact, supervisor summary, routing outcome and reporting payloads.

Everything here turns a finished reading run into the structures the rest of the
workflow consumes: the compact summary shown to Supervisor, the artifact holding
the full report and its evidence, the outcome the workflow layer routes on, and
the decision event that records the run.
"""

import json
from typing import cast
from uuid import UUID

from app.domain.papers import PdfDocument
from app.domain.rag import (
    CoverageMatrix,
    EvidenceLibrary,
    RetrievalStrategy,
    TreeRetrievalReport,
)
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    DecisionSource,
    ReaderAgentSummary,
    ReaderOutcome,
    ReaderTask,
    ReadingEvidenceAssessment,
    ReadingReport,
)


def compact_summary(value: str, max_length: int = 1_000) -> str:
    """Normalize and shorten a report summary to what Supervisor may read."""

    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3].rstrip()}..."


def unique_hit_count(reports: list[TreeRetrievalReport]) -> int:
    """Distinct retrieved chunks across every retrieval round."""

    return len({hit.node_id for report in reports for hit in report.hits})


def missing_paper_ids(reports: list[TreeRetrievalReport]) -> list[str]:
    """Papers named in the scope that no retrieval round could return."""

    return list(
        dict.fromkeys(
            paper_id
            for report in reports
            for paper_id in report.missing_paper_ids
        )
    )


def latest_strategy(reports: list[TreeRetrievalReport]) -> RetrievalStrategy | None:
    """Strategy of the most recent round that reported one."""

    return next(
        (
            report.strategy
            for report in reversed(reports)
            if report.strategy is not None
        ),
        None,
    )


def build_reader_artifact(
    *,
    task: ReaderTask,
    report: ReadingReport,
    library: EvidenceLibrary,
    coverage: CoverageMatrix | None,
    retrieval_reports: list[TreeRetrievalReport],
    pdf_document: PdfDocument | None,
    material_error: str | None,
    attempted_queries: list[str],
    retrieval_rounds: int,
    source_artifact_ids: list[UUID],
) -> AgentArtifact:
    """Wrap a reading report, its evidence and its coverage into one artifact."""

    return AgentArtifact(
        title=f"Reading report: {report.paper_or_material}",
        supervisor_summary=ReaderAgentSummary(
            summary=compact_summary(report.analysis_summary),
            paper_or_material=report.paper_or_material,
            objective_satisfied=report.objective_satisfied,
            answered_points=report.answered_points,
            blocking_gaps=report.blocking_gaps,
            evidence_scope=report.evidence_scope,
            limitations=report.limitations,
            pdf_truncated=pdf_document.truncated if pdf_document else False,
            pdf_error=material_error,
            retrieval_mode=(retrieval_reports[-1].mode if retrieval_reports else None),
            retrieval_strategy=latest_strategy(retrieval_reports),
            retrieval_hit_count=unique_hit_count(retrieval_reports),
            retrieval_rounds=retrieval_rounds,
            attempted_queries=attempted_queries,
            target_paper_ids=task.target_paper_ids,
            missing_paper_ids=missing_paper_ids(retrieval_reports),
            coverage=coverage,
        ),
        content=json.dumps(
            {
                "reading_report": report.model_dump(mode="json"),
                "evidence_library": library.model_dump(mode="json"),
                "coverage_matrix": (
                    coverage.model_dump(mode="json") if coverage is not None else None
                ),
            },
            ensure_ascii=False,
        ),
        source_artifact_ids=source_artifact_ids,
    )


def build_reader_outcome(
    *,
    task: ReaderTask,
    artifact: AgentArtifact,
    report: ReadingReport,
    assessment: ReadingEvidenceAssessment | None,
) -> ReaderOutcome:
    """Hand the workflow what it needs to route, without naming a next agent."""

    return ReaderOutcome(
        depth=task.depth,
        artifact_id=artifact.id,
        objective_satisfied=report.objective_satisfied,
        missing_requirements=(
            list(assessment.missing_requirements) if assessment is not None else []
        ),
    )


def build_reader_decision_payload(
    *,
    task: ReaderTask,
    artifact: AgentArtifact,
    report: ReadingReport,
    coverage: CoverageMatrix | None,
    pdf_document: PdfDocument | None,
    source_count: int,
) -> dict[str, JsonValue]:
    """Decision event for one finished reading run, read off its own artifact."""

    summary = artifact.supervisor_summary
    if not isinstance(summary, ReaderAgentSummary):
        raise ValueError("Reader decision payload requires a reading report artifact")
    strategy = summary.retrieval_strategy
    return {
        "source": DecisionSource.MODEL.value,
        "actor": "reader",
        "stage": "reader",
        "summary": summary.summary,
        "artifact_id": str(artifact.id),
        "source_count": source_count,
        "evidence_count": len(report.evidence),
        "objective_satisfied": report.objective_satisfied,
        "answered_points": cast(JsonValue, report.answered_points),
        "blocking_gaps": cast(JsonValue, report.blocking_gaps),
        "evidence_scope": report.evidence_scope,
        "limitations": cast(JsonValue, report.limitations),
        "pdf_pages": pdf_document.page_count if pdf_document else 0,
        "pdf_truncated": pdf_document.truncated if pdf_document else False,
        "retrieval_rounds": summary.retrieval_rounds,
        "attempted_queries": cast(JsonValue, summary.attempted_queries),
        "retrieval_hit_count": summary.retrieval_hit_count,
        "retrieval_strategy": strategy.value if strategy is not None else None,
        "target_paper_ids": cast(JsonValue, task.target_paper_ids),
        "missing_paper_ids": cast(JsonValue, summary.missing_paper_ids),
        "coverage": (
            cast(JsonValue, coverage.model_dump(mode="json"))
            if coverage is not None
            else None
        ),
        "coverage_ratio": coverage.coverage_ratio if coverage is not None else None,
        "candidate_ratio": coverage.candidate_ratio if coverage is not None else None,
        "candidate_cell_count": (
            coverage.candidate_cell_count if coverage is not None else None
        ),
        "verified_cell_count": (
            coverage.covered_cell_count if coverage is not None else None
        ),
        "unresolved_cells": cast(
            JsonValue,
            (
                [cell.model_dump(mode="json") for cell in coverage.unresolved_cells()]
                if coverage is not None
                else []
            ),
        ),
    }
