"""Run a labelled retrieval dataset and report whether the right evidence was found.

The metrics answer the acceptance question for multi-paper research: which expected
papers and which expected evidence were missed, and whether what came back is the
right evidence rather than merely evidence from the right paper. They are not a
claim about answer quality, and they are only as meaningful as the annotations they
run against.
"""

from collections.abc import Sequence
from time import perf_counter

from app.application.tree_retrieval import TreeRagRetriever
from app.domain.rag import RetrievalQuery, TreeRetrievalReport
from app.eval.dataset import (
    RetrievalEvalCase,
    RetrievalEvalCaseResult,
    RetrievalEvalDataset,
    RetrievalEvalReport,
)

__all__ = [
    "RetrievalEvalDataset",
    "RetrievalEvalReport",
    "RetrievalEvaluator",
    "load_dataset",
]


def load_dataset(payload: str) -> RetrievalEvalDataset:
    return RetrievalEvalDataset.model_validate_json(payload)


class RetrievalEvaluator:
    def __init__(self, retriever: TreeRagRetriever) -> None:
        self._retriever = retriever

    async def evaluate(self, dataset: RetrievalEvalDataset) -> RetrievalEvalReport:
        results = [await self._evaluate_case(case) for case in dataset.cases]
        answerable = [item for item in results if item.expects_answer]
        unanswerable = [item for item in results if not item.expects_answer]
        latencies = [item.latency_ms for item in results]
        return RetrievalEvalReport(
            dataset=dataset.name,
            case_count=len(results),
            answerable_case_count=len(answerable),
            unanswerable_case_count=len(unanswerable),
            target_paper_recall=_optional_mean(
                [item.target_paper_recall for item in answerable]
            ),
            evidence_recall=_optional_mean([item.evidence_recall for item in answerable]),
            evidence_precision=_optional_mean(
                [item.evidence_precision for item in answerable]
            ),
            paper_source_accuracy=_optional_mean(
                [item.paper_source_accuracy for item in answerable]
            ),
            candidate_dimension_coverage=_optional_mean(
                [item.candidate_dimension_coverage for item in answerable]
            ),
            dimension_support_rate=_optional_mean(
                [item.dimension_support_rate for item in answerable]
            ),
            strategy_accuracy=_optional_mean(
                [
                    float(item.strategy_matched)
                    for item in results
                    if item.strategy_matched is not None
                ]
            ),
            false_recall_rate=_optional_mean(
                [
                    float(item.false_recall)
                    for item in unanswerable
                    if item.false_recall is not None
                ]
            ),
            false_candidate_rate=_optional_mean(
                [
                    float(item.false_candidate)
                    for item in unanswerable
                    if item.false_candidate is not None
                ]
            ),
            mean_latency_ms=_mean(latencies),
            latency_p50_ms=_percentile(latencies, 0.50),
            latency_p95_ms=_percentile(latencies, 0.95),
            cases=results,
        )

    async def _evaluate_case(self, case: RetrievalEvalCase) -> RetrievalEvalCaseResult:
        queries = self._queries(case)
        started = perf_counter()
        report = await self._retriever.retrieve(
            case.question,
            paper_ids=case.paper_ids or None,
            mode=case.mode,
            strategy=case.strategy_override,
            queries=queries,
        )
        latency_ms = (perf_counter() - started) * 1000
        return self._case_result(case, report, latency_ms)

    @staticmethod
    def _queries(case: RetrievalEvalCase) -> list[RetrievalQuery] | None:
        if not case.sub_questions:
            return None
        return [
            RetrievalQuery(
                query=item.query,
                mode=item.mode or case.mode,
                paper_ids=list(item.paper_ids) or list(case.paper_ids),
                dimension=item.dimension,
            )
            for item in case.sub_questions
        ]

    @staticmethod
    def _case_result(
        case: RetrievalEvalCase,
        report: TreeRetrievalReport,
        latency_ms: float,
    ) -> RetrievalEvalCaseResult:
        matched_anchors = _anchor_matches(case, report)
        evidence_recall, unmatched_anchors = _evidence_recall(case, matched_anchors)
        evidence_precision = _evidence_precision(case, report, matched_anchors)
        dimension_support, unsupported_cells = _dimension_support(case, matched_anchors)
        candidate_coverage, unretrieved_cells = _candidate_dimension_coverage(case, report)
        returned_papers = {hit.paper_id for hit in report.hits}
        expected_papers = list(case.expected_paper_ids)
        matched_papers = [
            paper_id for paper_id in expected_papers if paper_id in returned_papers
        ]
        target_paper_recall = (
            len(matched_papers) / len(expected_papers) if expected_papers else None
        )
        strategy_matched = (
            None
            if case.expected_strategy is None
            else report.strategy is case.expected_strategy
        )
        candidate_cells = (
            report.coverage.candidate_cells() if report.coverage is not None else []
        )
        return RetrievalEvalCaseResult(
            case_id=case.case_id,
            strategy=report.strategy,
            strategy_matched=strategy_matched,
            expects_answer=case.expects_answer,
            target_paper_recall=target_paper_recall,
            evidence_recall=evidence_recall,
            evidence_precision=evidence_precision,
            paper_source_accuracy=_paper_source_accuracy(case, report),
            candidate_dimension_coverage=candidate_coverage,
            dimension_support_rate=dimension_support,
            hit_count=len(report.hits),
            keywords_recalled=report.keyword_candidate_count,
            keyword_truncated_terms=list(report.keyword_truncated_terms),
            missing_paper_ids=[
                paper_id for paper_id in expected_papers if paper_id not in returned_papers
            ],
            unretrieved_cells=unretrieved_cells,
            unsupported_cells=unsupported_cells,
            unmatched_anchors=unmatched_anchors,
            false_recall=(bool(report.hits) if not case.expects_answer else None),
            false_candidate=(bool(candidate_cells) if not case.expects_answer else None),
            latency_ms=latency_ms,
        )


def _anchor_matches(
    case: RetrievalEvalCase,
    report: TreeRetrievalReport,
) -> set[int]:
    """Indexes of expected anchors that appear in a returned chunk of their paper."""

    hits_by_paper: dict[str, list[str]] = {}
    for hit in report.hits:
        hits_by_paper.setdefault(hit.paper_id, []).append(hit.text.casefold())
    matched: set[int] = set()
    for index, anchor in enumerate(case.expected_evidence):
        texts = hits_by_paper.get(anchor.paper_id, [])
        if any(anchor.anchor.casefold() in text for text in texts):
            matched.add(index)
    return matched


def _evidence_recall(
    case: RetrievalEvalCase,
    matched_anchors: set[int],
) -> tuple[float | None, list[str]]:
    if not case.expected_evidence:
        return None, []
    unmatched = [
        f"{anchor.paper_id}:{anchor.anchor}"
        for index, anchor in enumerate(case.expected_evidence)
        if index not in matched_anchors
    ]
    total = len(case.expected_evidence)
    return (total - len(unmatched)) / total, unmatched


def _evidence_precision(
    case: RetrievalEvalCase,
    report: TreeRetrievalReport,
    matched_anchors: set[int],
) -> float | None:
    """Share of returned chunks that are an expected evidence fragment.

    A chunk counts as correct when it comes from the expected paper and contains one
    of that paper's anchors. Returning an unrelated paragraph from the right paper
    lowers this, which is what ``paper_source_accuracy`` alone cannot see.
    """

    if not case.expected_evidence:
        return None
    if not report.hits:
        return 0.0
    anchors_by_paper: dict[str, list[str]] = {}
    for index, anchor in enumerate(case.expected_evidence):
        if index not in matched_anchors:
            continue
        anchors_by_paper.setdefault(anchor.paper_id, []).append(anchor.anchor.casefold())
    correct = 0
    for hit in report.hits:
        text = hit.text.casefold()
        if any(anchor in text for anchor in anchors_by_paper.get(hit.paper_id, [])):
            correct += 1
    return correct / len(report.hits)


def _paper_source_accuracy(
    case: RetrievalEvalCase,
    report: TreeRetrievalReport,
) -> float | None:
    """Share of returned chunks that come from an expected paper.

    Source accuracy, not evidence correctness: a chunk from the right paper that
    answers nothing still counts here. Returns None when the case declares no
    expected papers, instead of reporting a misleading perfect score.
    """

    expected = set(case.expected_paper_ids)
    if not expected:
        return None
    if not report.hits:
        return 0.0
    return len([hit for hit in report.hits if hit.paper_id in expected]) / len(report.hits)


def _dimension_support(
    case: RetrievalEvalCase,
    matched_anchors: set[int],
) -> tuple[float | None, list[str]]:
    """Share of anchored paper x dimension cells with a matching evidence chunk."""

    anchored: dict[tuple[str, str], list[int]] = {}
    for index, anchor in enumerate(case.expected_evidence):
        if anchor.dimension is None:
            continue
        anchored.setdefault((anchor.paper_id, anchor.dimension), []).append(index)
    if not anchored:
        return None, []
    unsupported = [
        f"{paper_id}:{dimension}"
        for (paper_id, dimension), indexes in anchored.items()
        if not any(index in matched_anchors for index in indexes)
    ]
    return (len(anchored) - len(unsupported)) / len(anchored), unsupported


def _candidate_dimension_coverage(
    case: RetrievalEvalCase,
    report: TreeRetrievalReport,
) -> tuple[float | None, list[str]]:
    """Fraction of expected cells that received a candidate chunk.

    This is a retrieval-side recall measure. Whether a candidate answers its
    dimension is decided by an evidence reader, so the harness deliberately does not
    report a verified-coverage number here.
    """

    dimensions = list(case.expected_dimensions)
    if not dimensions or not case.expected_paper_ids:
        return None, []
    unretrieved: list[str] = []
    for paper_id in case.expected_paper_ids:
        for dimension in dimensions:
            cell = (
                None
                if report.coverage is None
                else report.coverage.cell_for(paper_id, dimension)
            )
            if cell is None or not cell.evidence_ids:
                unretrieved.append(f"{paper_id}:{dimension}")
    total = len(dimensions) * len(case.expected_paper_ids)
    return (total - len(unretrieved)) / total, unretrieved


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_mean(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return _mean(present) if present else None


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
