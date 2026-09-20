"""State contract of the Reader subgraph."""

from dataclasses import dataclass
from typing import Literal

from typing_extensions import TypedDict

from app.domain.papers import Paper, PdfDocument
from app.domain.rag import TreeRetrievalReport
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.models import (
    CoverageGapJudgment,
    ReaderTask,
    ReadingEvidenceAssessment,
    ReadingPlan,
    ReadingReport,
    ResearchPlan,
)


@dataclass(frozen=True, slots=True)
class PdfCandidate:
    paper: Paper
    source_artifact_id: str


class ReaderState(TypedDict):
    task: ReaderTask
    research_plan: ResearchPlan | None
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
