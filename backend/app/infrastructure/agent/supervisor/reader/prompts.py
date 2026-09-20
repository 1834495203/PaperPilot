"""Model input assembly for the Reader stages.

Each builder is a pure function of the reader state and the already-computed
evidence library and coverage matrix, so the exact text sent to the model can be
inspected and diffed without running the graph.
"""

import json
from typing import cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.domain.rag import CoverageMatrix, EvidenceLibrary, RetrievalMode
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.prompts import (
    READER_EVIDENCE_PROMPT,
    READER_PLANNING_PROMPT,
    READER_PROMPT,
)
from app.infrastructure.agent.supervisor.reader.state import ReaderState


def reading_material(state: ReaderState, library: EvidenceLibrary) -> str:
    """Render the material a stage may read, together with the citable evidence.

    Only material a stage can reason about is rendered. Retrieval diagnostics such
    as reranker state and candidate-pool sizes stay in the tool events, where they
    are auditable, instead of consuming the model's input budget.
    """

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
                    }
                    for report in state["retrieval_reports"]
                ],
                "evidence_library": library.model_dump(mode="json"),
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
                "evidence_library": library.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
    material = json.dumps(
        {
            "paper_metadata": state["local_metadata"],
            "paper": candidate.paper.model_dump(mode="json") if candidate else None,
            "evidence_library": library.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
    return (
        f"{material}\n\nMaterial retrieval failed: {state['material_error']}"
        if state["material_error"]
        else material
    )


def build_planning_messages(state: ReaderState) -> list[BaseMessage]:
    """Ask the planning stage for a retrieval strategy and sub-questions."""

    task = state["task"]
    targets = task.target_paper_ids
    return [
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
    ]


def build_assessment_messages(
    state: ReaderState,
    library: EvidenceLibrary,
    coverage: CoverageMatrix | None,
    *,
    can_retry: bool,
) -> list[BaseMessage]:
    """Ask the evidence judge to settle the coverage cells it can see."""

    plan = state["plan"]
    if plan is None:
        raise ValueError("Reader evidence assessment requires a reading plan")
    return [
        SystemMessage(content=READER_EVIDENCE_PROMPT),
        HumanMessage(
            content=json.dumps(
                {
                    "reading_objective": state["task"].objective,
                    "reader_depth": state["task"].depth.value,
                    "retrieval_can_retry": can_retry,
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
                    "available_material": reading_material(state, library),
                    "material_error": state["material_error"],
                },
                ensure_ascii=False,
            )
        ),
    ]


def build_synthesis_messages(
    state: ReaderState,
    library: EvidenceLibrary,
    coverage: CoverageMatrix | None,
) -> list[BaseMessage]:
    """Ask the reading stage to report only what the supplied evidence states."""

    task = state["task"]
    scope = (
        "all locally indexed papers"
        if not task.target_paper_ids
        else task.target_paper_ids
    )
    prompt = (
        f"Assigned evidence-reading task:\n{task.objective}\n\n"
        f"Reader depth:\n{task.depth.value}\n\n"
        f"Retrieval scope:\n{scope}\n\n"
        f"Available material:\n{reading_material(state, library)}\n\n"
        f"Coverage matrix:\n"
        f"{coverage.model_dump_json() if coverage is not None else 'not available'}\n\n"
        "Analyze only the supplied evidence. Do not evaluate the overall workflow. "
        "Report each paper and dimension cell that stayed uncovered, and state plainly when "
        "the paper does not address a dimension instead of treating it as missing evidence."
    )
    return [
        SystemMessage(content=READER_PROMPT),
        HumanMessage(content=prompt),
    ]
