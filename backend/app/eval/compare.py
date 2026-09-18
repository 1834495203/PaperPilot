"""Compare two retrieval evaluation runs case by case.

This is how a corpus change is measured: run the same labelled dataset against the
library before and after adding distractor papers, an embedding change, or a
retrieval setting change, then diff the reports. The aggregate delta says whether
the change helped; the per-case delta says which question lost its evidence, which
is what you need when deciding whether a regression is acceptable.
"""

from pydantic import BaseModel, ConfigDict, Field

from app.eval.dataset import RetrievalEvalCaseResult, RetrievalEvalReport

_METRICS = (
    "target_paper_recall",
    "evidence_recall",
    "evidence_precision",
    "paper_source_accuracy",
    "candidate_dimension_coverage",
    "dimension_support_rate",
    "strategy_accuracy",
    "false_recall_rate",
    "false_candidate_rate",
    "mean_latency_ms",
    "latency_p50_ms",
    "latency_p95_ms",
)


class MetricDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric: str
    baseline: float | None
    current: float | None

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline


class CaseDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    baseline_hit_count: int = Field(ge=0)
    current_hit_count: int = Field(ge=0)
    baseline_paper_recall: float | None = None
    current_paper_recall: float | None = None
    baseline_evidence_recall: float | None = None
    current_evidence_recall: float | None = None
    lost_missing_paper_ids: list[str] = Field(default_factory=list)
    gained_missing_paper_ids: list[str] = Field(default_factory=list)
    lost_anchors: list[str] = Field(default_factory=list)
    gained_anchors: list[str] = Field(default_factory=list)
    regression: bool = False
    improvement: bool = False


class RetrievalEvalComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_dataset: str
    current_dataset: str
    metrics: list[MetricDelta]
    cases: list[CaseDelta]
    regressed_case_ids: list[str] = Field(default_factory=list)
    improved_case_ids: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"# Retrieval comparison: {self.current_dataset} vs {self.baseline_dataset}",
            "",
            "| metric | baseline | current | delta |",
            "|---|---|---|---|",
        ]
        for item in self.metrics:
            lines.append(
                f"| {item.metric} | {_format(item.baseline)} | {_format(item.current)} | "
                f"{_format_signed(item.delta)} |"
            )
        lines.extend(
            [
                "",
                "## Per case",
                "",
                "| case | hits | paper recall | evidence recall | lost |",
                "|---|---|---|---|---|",
            ]
        )
        for case in self.cases:
            lost = ", ".join(
                [*case.gained_missing_paper_ids, *case.lost_anchors]
            ) or "-"
            lines.append(
                f"| {case.case_id} | {case.baseline_hit_count}→{case.current_hit_count} | "
                f"{_format(case.baseline_paper_recall)}→"
                f"{_format(case.current_paper_recall)} | "
                f"{_format(case.baseline_evidence_recall)}→"
                f"{_format(case.current_evidence_recall)} | {lost} |"
            )
        lines.append("")
        lines.append(
            f"regressed: {len(self.regressed_case_ids)} "
            f"({', '.join(self.regressed_case_ids) or '-'})"
        )
        lines.append(
            f"improved: {len(self.improved_case_ids)} "
            f"({', '.join(self.improved_case_ids) or '-'})"
        )
        return "\n".join(lines) + "\n"


def compare_reports(
    baseline: RetrievalEvalReport,
    current: RetrievalEvalReport,
) -> RetrievalEvalComparison:
    baseline_cases = {case.case_id: case for case in baseline.cases}
    current_cases = {case.case_id: case for case in current.cases}
    cases: list[CaseDelta] = []
    regressed: list[str] = []
    improved: list[str] = []
    for case_id, current_case in current_cases.items():
        baseline_case = baseline_cases.get(case_id)
        if baseline_case is None:
            continue
        delta = _case_delta(baseline_case, current_case)
        cases.append(delta)
        if delta.regression:
            regressed.append(case_id)
        if delta.improvement:
            improved.append(case_id)
    return RetrievalEvalComparison(
        baseline_dataset=baseline.dataset,
        current_dataset=current.dataset,
        metrics=[
            MetricDelta(
                metric=metric,
                baseline=_metric_value(baseline, metric),
                current=_metric_value(current, metric),
            )
            for metric in _METRICS
        ],
        cases=cases,
        regressed_case_ids=regressed,
        improved_case_ids=improved,
    )


def _metric_value(report: RetrievalEvalReport, metric: str) -> float | None:
    value = getattr(report, metric)
    if value is None:
        return None
    return float(value)


def _case_delta(
    baseline: RetrievalEvalCaseResult,
    current: RetrievalEvalCaseResult,
) -> CaseDelta:
    baseline_missing = set(baseline.missing_paper_ids)
    current_missing = set(current.missing_paper_ids)
    baseline_unmatched = set(baseline.unmatched_anchors)
    current_unmatched = set(current.unmatched_anchors)
    gained_missing = sorted(current_missing - baseline_missing)
    lost_anchors = sorted(current_unmatched - baseline_unmatched)
    lost_missing = sorted(baseline_missing - current_missing)
    gained_anchors = sorted(baseline_unmatched - current_unmatched)
    regression = bool(gained_missing or lost_anchors) or _lower(
        baseline.target_paper_recall, current.target_paper_recall
    ) or _lower(baseline.evidence_recall, current.evidence_recall)
    improvement = bool(lost_missing or gained_anchors) or _higher(
        baseline.target_paper_recall, current.target_paper_recall
    ) or _higher(baseline.evidence_recall, current.evidence_recall)
    return CaseDelta(
        case_id=current.case_id,
        baseline_hit_count=baseline.hit_count,
        current_hit_count=current.hit_count,
        baseline_paper_recall=baseline.target_paper_recall,
        current_paper_recall=current.target_paper_recall,
        baseline_evidence_recall=baseline.evidence_recall,
        current_evidence_recall=current.evidence_recall,
        lost_missing_paper_ids=lost_missing,
        gained_missing_paper_ids=gained_missing,
        lost_anchors=lost_anchors,
        gained_anchors=gained_anchors,
        regression=regression,
        improvement=improvement and not regression,
    )


def _lower(baseline: float | None, current: float | None) -> bool:
    if baseline is None or current is None:
        return False
    return current < baseline


def _higher(baseline: float | None, current: float | None) -> bool:
    if baseline is None or current is None:
        return False
    return current > baseline


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _format_signed(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"
