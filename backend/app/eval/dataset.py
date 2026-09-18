"""Retrieval evaluation models: a labelled question set and its acceptance metrics."""

from pydantic import BaseModel, ConfigDict, Field

from app.domain.rag import RetrievalMode, RetrievalStrategy


class ExpectedEvidence(BaseModel):
    """One evidence anchor that a correct retrieval must return for a paper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str = Field(min_length=1)
    anchor: str = Field(
        min_length=3,
        description="Case-folded substring that must appear in a returned chunk",
    )
    dimension: str | None = Field(default=None, max_length=100)


class EvalSubQuestion(BaseModel):
    """A dimension-scoped sub-question the evaluator sends to the retriever."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=2_000)
    dimension: str = Field(min_length=1, max_length=100)
    paper_ids: list[str] = Field(default_factory=list)
    mode: RetrievalMode | None = None


class RetrievalEvalCase(BaseModel):
    """One labelled retrieval task.

    ``expected_paper_ids`` and ``expected_evidence`` are the annotations that make
    the metrics meaningful: target-paper recall and evidence recall measure what a
    correct answer must not miss, while evidence precision and dimension support
    measure whether the returned chunks are the *right* evidence rather than merely
    coming from the right paper.

    Dimension support only counts for cells that carry an anchored expectation, and
    dimension coverage only counts when the case asks for those dimensions, which
    means supplying ``sub_questions`` (one per paper x dimension). A case that
    declares ``expected_dimensions`` without asking for them reports those cells as
    unretrieved, because retrieval was never told to look for them.

    Set ``expects_answer=false`` for a question the corpus cannot answer; those
    cases feed the false-recall and false-candidate rates instead of the recall
    metrics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    mode: RetrievalMode = RetrievalMode.METHOD
    paper_ids: list[str] = Field(default_factory=list)
    strategy_override: RetrievalStrategy | None = None
    expected_strategy: RetrievalStrategy | None = None
    expected_paper_ids: list[str] = Field(default_factory=list)
    expected_dimensions: list[str] = Field(default_factory=list)
    expected_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    sub_questions: list[EvalSubQuestion] = Field(default_factory=list)
    expects_answer: bool = True
    notes: str | None = None


class RetrievalEvalDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    collection: str | None = None
    cases: list[RetrievalEvalCase] = Field(min_length=1)


class RetrievalEvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    strategy: RetrievalStrategy
    strategy_matched: bool | None
    expects_answer: bool
    target_paper_recall: float | None = Field(default=None, ge=0, le=1)
    evidence_recall: float | None = Field(default=None, ge=0, le=1)
    evidence_precision: float | None = Field(default=None, ge=0, le=1)
    paper_source_accuracy: float | None = Field(default=None, ge=0, le=1)
    candidate_dimension_coverage: float | None = Field(default=None, ge=0, le=1)
    dimension_support_rate: float | None = Field(default=None, ge=0, le=1)
    hit_count: int = Field(ge=0)
    keywords_recalled: int = Field(default=0, ge=0)
    keyword_truncated_terms: list[str] = Field(default_factory=list)
    missing_paper_ids: list[str] = Field(default_factory=list)
    unretrieved_cells: list[str] = Field(default_factory=list)
    unsupported_cells: list[str] = Field(default_factory=list)
    unmatched_anchors: list[str] = Field(default_factory=list)
    false_recall: bool | None = None
    false_candidate: bool | None = None
    latency_ms: float = Field(ge=0)


class RetrievalEvalReport(BaseModel):
    """Aggregate metrics for one dataset run.

    Recall-style metrics answer "did the expected paper and evidence survive
    retrieval". ``paper_source_accuracy`` and ``evidence_precision`` answer "is what
    came back actually the right evidence" - coming from the expected paper is not
    enough. ``candidate_dimension_coverage`` is a retrieval-side measure of whether
    each expected paper x dimension cell received a candidate; whether a cell is
    answered needs an evidence reader, so this harness never reports a verified
    coverage number.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    case_count: int = Field(ge=1)
    answerable_case_count: int = Field(default=0, ge=0)
    unanswerable_case_count: int = Field(default=0, ge=0)
    target_paper_recall: float | None = Field(default=None, ge=0, le=1)
    evidence_recall: float | None = Field(default=None, ge=0, le=1)
    evidence_precision: float | None = Field(default=None, ge=0, le=1)
    paper_source_accuracy: float | None = Field(default=None, ge=0, le=1)
    candidate_dimension_coverage: float | None = Field(default=None, ge=0, le=1)
    dimension_support_rate: float | None = Field(default=None, ge=0, le=1)
    strategy_accuracy: float | None = Field(default=None, ge=0, le=1)
    false_recall_rate: float | None = Field(default=None, ge=0, le=1)
    false_candidate_rate: float | None = Field(default=None, ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    latency_p50_ms: float = Field(default=0.0, ge=0)
    latency_p95_ms: float = Field(default=0.0, ge=0)
    cases: list[RetrievalEvalCaseResult]

    def to_markdown(self) -> str:
        lines = [
            f"# Retrieval evaluation: {self.dataset}",
            "",
            f"cases: {self.case_count} "
            f"(answerable {self.answerable_case_count}, "
            f"unanswerable {self.unanswerable_case_count})",
            "",
            "| metric | value |",
            "|---|---|",
            f"| target paper recall | {_format_optional(self.target_paper_recall)} |",
            f"| evidence recall | {_format_optional(self.evidence_recall)} |",
            f"| evidence precision | {_format_optional(self.evidence_precision)} |",
            f"| paper source accuracy | {_format_optional(self.paper_source_accuracy)} |",
            "| candidate dimension coverage | "
            f"{_format_optional(self.candidate_dimension_coverage)} |",
            f"| dimension support rate | {_format_optional(self.dimension_support_rate)} |",
            f"| strategy accuracy | {_format_optional(self.strategy_accuracy)} |",
            f"| false recall rate | {_format_optional(self.false_recall_rate)} |",
            f"| false candidate rate | {_format_optional(self.false_candidate_rate)} |",
            f"| mean latency (ms) | {self.mean_latency_ms:.1f} |",
            f"| p50 / p95 latency (ms) | {self.latency_p50_ms:.1f} / "
            f"{self.latency_p95_ms:.1f} |",
            "",
            "| case | strategy | paper recall | evidence recall | evidence precision |"
            " dimension support | missing |",
            "|---|---|---|---|---|---|---|",
        ]
        for case in self.cases:
            lines.append(
                f"| {case.case_id} | {case.strategy.value} | "
                f"{_format_optional(case.target_paper_recall)} | "
                f"{_format_optional(case.evidence_recall)} | "
                f"{_format_optional(case.evidence_precision)} | "
                f"{_format_optional(case.dimension_support_rate)} | "
                f"{', '.join(case.missing_paper_ids) or '-'} |"
            )
        return "\n".join(lines) + "\n"


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
