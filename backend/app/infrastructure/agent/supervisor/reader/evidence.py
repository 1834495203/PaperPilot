"""Evidence library, coverage merging and reference validation.

Every rule here is deterministic: no model is called and no state is mutated, so
the evidence a report may cite and the coverage it may claim can be verified on
its own. Functions take explicit arguments instead of a reader state object so
they stay reusable by any caller that holds the same material.
"""

import hashlib
import json

from app.domain.papers import PdfDocument
from app.domain.rag import (
    CoverageCell,
    CoverageMatrix,
    CoverageStatus,
    Evidence,
    EvidenceLibrary,
    TreeRetrievalReport,
)
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.models import (
    CoverageGapJudgment,
    ReaderTask,
    ReadingPlan,
    ReadingReport,
)
from app.infrastructure.agent.supervisor.reader.state import PdfCandidate


def evidence_id(chunk_id: str) -> str:
    """Stable Evidence Library ID for one retrievable chunk."""

    return f"E-{hashlib.sha256(chunk_id.encode('utf-8')).hexdigest()[:12]}"


def build_evidence_library(
    task: ReaderTask,
    retrieval_reports: list[TreeRetrievalReport],
    pdf_document: PdfDocument | None,
    pdf_candidate: PdfCandidate | None,
    local_metadata: dict[str, JsonValue] | None,
) -> EvidenceLibrary:
    """Collect citable evidence, falling back to the raw material that was read.

    Retrieved chunks are the primary evidence. When retrieval produced nothing the
    extracted PDF text, and after that the authoritative metadata, is registered as
    the single citable passage, so a report is never left with nothing to cite.
    """

    evidence: list[Evidence] = []
    seen_chunks: set[str] = set()
    for report in retrieval_reports:
        for hit in report.hits:
            if hit.node_id in seen_chunks:
                continue
            seen_chunks.add(hit.node_id)
            evidence.append(
                Evidence(
                    evidence_id=evidence_id(hit.node_id),
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
    if not evidence and pdf_document is not None:
        paper_id = pdf_candidate.paper.paper_id if pdf_candidate is not None else "external-paper"
        title = pdf_candidate.paper.title if pdf_candidate is not None else paper_id
        evidence.append(
            Evidence(
                evidence_id=evidence_id(f"{paper_id}:pdf-extraction"),
                paper_id=paper_id,
                paper_title=title,
                chunk_id=f"{paper_id}:pdf-extraction",
                section_path=[],
                page_start=1,
                page_end=pdf_document.extracted_pages or None,
                raw_text=pdf_document.text,
                evidence_text=pdf_document.text,
                retrieval_score=1.0,
            )
        )
    if not evidence and local_metadata is not None:
        metadata_text = json.dumps(local_metadata, ensure_ascii=False)
        paper_id = task.target_paper_ids[0] if task.target_paper_ids else "local-metadata"
        title_value = local_metadata.get("title")
        if not isinstance(title_value, str):
            title_value = next(
                (
                    item.get("title")
                    for item in local_metadata.values()
                    if isinstance(item, dict)
                ),
                None,
            )
        title = title_value if isinstance(title_value, str) else paper_id
        evidence.append(
            Evidence(
                evidence_id=evidence_id(f"{paper_id}:metadata"),
                paper_id=paper_id,
                paper_title=title,
                chunk_id=f"{paper_id}:metadata",
                section_path=[],
                raw_text=metadata_text,
                evidence_text=metadata_text,
                retrieval_score=1.0,
            )
        )
    return EvidenceLibrary(objective=task.objective, evidence=evidence)


def merge_coverage(
    retrieval_reports: list[TreeRetrievalReport],
    plan: ReadingPlan | None,
    judgments: list[CoverageGapJudgment],
    library: EvidenceLibrary,
) -> CoverageMatrix | None:
    """Merge every retrieval round with the judge verdicts into one matrix."""

    if not retrieval_reports:
        return None
    paper_ids: list[str] = []
    dimensions: list[str] = []
    candidates: dict[tuple[str, str], list[str]] = {}
    for report in retrieval_reports:
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
    coverage = CoverageMatrix.build(
        paper_ids=paper_ids,
        dimensions=preferred or dimensions,
    )
    for (paper_id, dimension), cell_evidence_ids in candidates.items():
        if coverage.cell_for(paper_id, dimension) is None:
            continue
        coverage = coverage.with_cell(
            CoverageCell(
                paper_id=paper_id,
                dimension=dimension,
                status=CoverageStatus.CANDIDATE,
                evidence_ids=cell_evidence_ids,
            )
        )
    return apply_judgments(coverage, judgments, library=library)


def apply_judgments(
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
        candidate_ids = existing_cell.evidence_ids if existing_cell is not None else []
        status = CoverageStatus(judgment.status)
        note = judgment.reason
        verified_ids: list[str] = []
        if status is CoverageStatus.COVERED:
            verified_ids = [
                cited_id
                for cited_id in judgment.evidence_ids
                if evidence_paper_ids.get(cited_id) == judgment.paper_id
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


def merge_judgments(
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


def validate_report_references(
    report: ReadingReport,
    library: EvidenceLibrary,
) -> ReadingReport:
    """Drop evidence entries the library does not define, and say that we did."""

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
