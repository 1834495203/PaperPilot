import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.domain.rag import (
    CoverageCell,
    CoverageMatrix,
    CoverageStatus,
    RetrievalHit,
    RetrievalMode,
    RetrievalQuery,
    RetrievalSource,
    RetrievalStrategy,
    TreeRetrievalReport,
)
from app.eval.dataset import (
    RetrievalEvalCaseResult,
    RetrievalEvalDataset,
    RetrievalEvalReport,
)
from app.eval.runner import RetrievalEvaluator, load_dataset

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_eval.sample.json"


def _coverage_matrix(
    *,
    paper_ids: list[str],
    dimensions: list[str],
    hits: list[RetrievalHit],
) -> CoverageMatrix:
    """Mirror the retriever contract: delivered chunks become candidate cells."""

    matrix = CoverageMatrix.build(paper_ids=paper_ids, dimensions=dimensions)
    for hit in hits:
        for dimension in hit.matched_dimensions or [CoverageMatrix.DEFAULT_DIMENSION]:
            cell = matrix.cell_for(hit.paper_id, dimension)
            if cell is None:
                continue
            matrix = matrix.with_cell(
                CoverageCell(
                    paper_id=cell.paper_id,
                    dimension=cell.dimension,
                    status=CoverageStatus.CANDIDATE,
                    evidence_ids=[*cell.evidence_ids, hit.node_id],
                )
            )
    return matrix


class _StubRetriever:
    """Returns per-paper hits and records the queries the evaluator asked for."""

    def __init__(self, hits_by_query: dict[str, list[str]], *, strategy: RetrievalStrategy):
        self._hits_by_query = hits_by_query
        self._strategy = strategy
        self.queries: list[list[RetrievalQuery] | None] = []
        self.paper_ids: list[list[str] | None] = []

    async def retrieve(
        self,
        query: str,
        *,
        paper_ids: list[str] | None = None,
        mode: RetrievalMode,
        strategy: RetrievalStrategy | None = None,
        queries: list[RetrievalQuery] | None = None,
        budget: object | None = None,
    ) -> TreeRetrievalReport:
        del mode, strategy, budget
        self.queries.append(queries)
        self.paper_ids.append(paper_ids)
        plan = queries or [
            RetrievalQuery(query=query, mode=RetrievalMode.METHOD, paper_ids=paper_ids or [])
        ]
        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for item in plan:
            for entry in self._hits_by_query.get(item.query, []):
                paper_id, _, text = entry.partition("|")
                node_id = f"{paper_id}:{abs(hash(text)) % 10_000}"
                if node_id in seen:
                    continue
                seen.add(node_id)
                hits.append(
                    RetrievalHit(
                        rank=len(hits) + 1,
                        node_id=node_id,
                        paper_id=paper_id,
                        paper_title=paper_id,
                        section_path=["3 Method"],
                        text=text,
                        vector_score=0.9,
                        ranking_score=0.9,
                        matched_dimensions=(
                            [item.dimension] if item.dimension is not None else []
                        ),
                        matched_queries=[item.query],
                        source=RetrievalSource.VECTOR,
                    )
                )
        return TreeRetrievalReport(
            query=query,
            mode=RetrievalMode.METHOD,
            paper_ids=paper_ids or [],
            strategy=self._strategy,
            candidate_paper_ids=[hit.paper_id for hit in hits],
            coverage=_coverage_matrix(
                # Mirrors the retriever: a global query rows the papers it reached.
                paper_ids=list(paper_ids or [])
                or list(dict.fromkeys(hit.paper_id for hit in hits)),
                dimensions=[
                    item.dimension for item in plan if item.dimension is not None
                ],
                hits=hits,
            ),
            initial_hit_count=len(hits),
            expanded_candidate_count=0,
            hits=hits,
        )


def test_sample_dataset_matches_the_documented_format() -> None:
    dataset = load_dataset(FIXTURE.read_text(encoding="utf-8"))

    assert dataset.name == "paperpilot-retrieval-sample"
    assert [case.case_id for case in dataset.cases] == [
        "single-paper-fact",
        "multi-paper-comparison",
        "corpus-survey",
        "unanswerable-topic",
    ]
    comparison = dataset.cases[1]
    assert comparison.expected_strategy is RetrievalStrategy.MULTI_PAPER
    assert len(comparison.sub_questions) == 6
    assert all(anchor.dimension for anchor in comparison.expected_evidence)
    assert dataset.cases[3].expects_answer is False


@pytest.mark.asyncio
async def test_evaluator_scores_expected_papers_evidence_and_dimensions() -> None:
    dataset = RetrievalEvalDataset.model_validate(
        {
            "name": "unit",
            "cases": [
                {
                    "case_id": "compare",
                    "question": "Compare method and results",
                    "mode": "compare",
                    "paper_ids": ["paper-a", "paper-b"],
                    "expected_strategy": "multi_paper",
                    "expected_paper_ids": ["paper-a", "paper-b"],
                    "expected_dimensions": ["method", "results"],
                    "sub_questions": [
                        {
                            "query": "method of paper-a",
                            "dimension": "method",
                            "paper_ids": ["paper-a"],
                        },
                        {
                            "query": "results of paper-b",
                            "dimension": "results",
                            "paper_ids": ["paper-b"],
                        },
                    ],
                    "expected_evidence": [
                        {"paper_id": "paper-a", "anchor": "hierarchical"},
                        {"paper_id": "paper-b", "anchor": "latency"},
                    ],
                }
            ],
        }
    )
    retriever = _StubRetriever(
        {
            "method of paper-a": ["paper-a|We build a hierarchical index."],
            "results of paper-b": ["paper-b|Latency drops by half."],
        },
        strategy=RetrievalStrategy.MULTI_PAPER,
    )

    report = await RetrievalEvaluator(retriever).evaluate(dataset)  # type: ignore[arg-type]

    assert report.target_paper_recall == 1.0
    assert report.evidence_recall == 1.0
    assert report.evidence_precision == 1.0
    assert report.paper_source_accuracy == 1.0
    assert report.candidate_dimension_coverage == pytest.approx(0.5)
    assert report.strategy_accuracy == 1.0
    case = report.cases[0]
    assert case.missing_paper_ids == []
    assert case.unmatched_anchors == []
    assert set(case.unretrieved_cells) == {"paper-a:results", "paper-b:method"}


@pytest.mark.asyncio
async def test_evaluator_reports_missing_papers_evidence_and_cells() -> None:
    dataset = RetrievalEvalDataset.model_validate(
        {
            "name": "unit-misses",
            "cases": [
                {
                    "case_id": "compare",
                    "question": "Compare everything",
                    "mode": "compare",
                    "paper_ids": ["paper-a", "paper-b", "paper-c"],
                    "expected_strategy": "multi_paper",
                    "expected_paper_ids": ["paper-a", "paper-b", "paper-c"],
                    "expected_dimensions": ["method", "results"],
                    "sub_questions": [
                        {
                            "query": "method everywhere",
                            "dimension": "method",
                        }
                    ],
                    "expected_evidence": [
                        {"paper_id": "paper-a", "anchor": "hierarchical"},
                        {"paper_id": "paper-c", "anchor": "limitations"},
                    ],
                }
            ],
        }
    )
    retriever = _StubRetriever(
        {
            "method everywhere": [
                "paper-a|We build a hierarchical index.",
                "paper-b|This paragraph only discusses page formatting.",
            ]
        },
        strategy=RetrievalStrategy.SINGLE_PAPER,
    )

    report = await RetrievalEvaluator(retriever).evaluate(dataset)  # type: ignore[arg-type]

    assert report.target_paper_recall == pytest.approx(2 / 3)
    assert report.evidence_recall == pytest.approx(0.5)
    # Both returned chunks come from an expected paper, but only one is the
    # expected evidence, so source accuracy and evidence precision diverge.
    assert report.paper_source_accuracy == 1.0
    assert report.evidence_precision == pytest.approx(0.5)
    assert report.strategy_accuracy == 0.0
    case = report.cases[0]
    assert case.missing_paper_ids == ["paper-c"]
    assert case.unmatched_anchors == ["paper-c:limitations"]
    assert set(case.unretrieved_cells) == {
        "paper-a:results",
        "paper-b:results",
        "paper-c:method",
        "paper-c:results",
    }


@pytest.mark.asyncio
async def test_evaluator_passes_paper_scope_and_sub_questions_to_the_retriever() -> None:
    dataset = load_dataset(FIXTURE.read_text(encoding="utf-8")).model_copy(
        update={"cases": []}
    )
    case = load_dataset(FIXTURE.read_text(encoding="utf-8")).cases[1]
    dataset = RetrievalEvalDataset(name="scope", cases=[case])
    retriever = _StubRetriever({}, strategy=RetrievalStrategy.MULTI_PAPER)

    await RetrievalEvaluator(retriever).evaluate(dataset)  # type: ignore[arg-type]

    assert retriever.paper_ids == [["paper-a", "paper-b", "paper-c"]]
    queries = retriever.queries[0]
    assert queries is not None
    assert len(queries) == 6
    assert queries[0].dimension == "method"
    assert queries[0].paper_ids == ["paper-a"]


def test_report_renders_a_markdown_summary() -> None:
    report = RetrievalEvalReport(
        dataset="unit",
        case_count=1,
        target_paper_recall=0.5,
        evidence_recall=None,
        candidate_dimension_coverage=None,
        paper_source_accuracy=1.0,
        evidence_precision=None,
        strategy_accuracy=None,
        mean_latency_ms=12.5,
        cases=[],
    )

    markdown = report.to_markdown()

    assert "# Retrieval evaluation: unit" in markdown
    assert "| target paper recall | 0.500 |" in markdown
    assert "| evidence recall | n/a |" in markdown
    assert "| candidate dimension coverage | n/a |" in markdown


def test_coverage_matrix_helpers_are_serialisable() -> None:
    matrix = CoverageMatrix.build(paper_ids=["paper-a", "paper-b"], dimensions=["method"])
    matrix = matrix.with_cell(
        CoverageCell(
            paper_id="paper-a",
            dimension="method",
            status=CoverageStatus.NOT_STATED,
            note="No limitations section exists",
        )
    )

    payload = json.loads(matrix.model_dump_json())

    assert payload["covered_cell_count"] == 0
    assert payload["unresolved_cell_count"] == 2
    assert payload["paper_ids"] == ["paper-a", "paper-b"]
    assert payload["coverage_ratio"] == 0.0
    assert payload["cells"][0]["note"] == "No limitations section exists"


def _unused_sequence(values: Sequence[str]) -> int:
    return len(values)


@pytest.mark.asyncio
async def test_unanswerable_case_reports_false_recall_instead_of_recall() -> None:
    dataset = RetrievalEvalDataset.model_validate(
        {
            "name": "no-answer",
            "cases": [
                {
                    "case_id": "absent-topic",
                    "question": "Which paper reports a negative transfer result?",
                    "mode": "fact",
                    "expects_answer": False,
                },
                {
                    "case_id": "absent-topic-clean",
                    "question": "Which paper reports a quantum advantage?",
                    "mode": "fact",
                    "expects_answer": False,
                },
            ],
        }
    )
    retriever = _StubRetriever(
        {"Which paper reports a negative transfer result?": ["paper-a|Unrelated text"]},
        strategy=RetrievalStrategy.SINGLE_PAPER,
    )

    report = await RetrievalEvaluator(retriever).evaluate(dataset)  # type: ignore[arg-type]

    assert report.unanswerable_case_count == 2
    assert report.answerable_case_count == 0
    # Recall-style metrics stay undefined when no case declares an answer.
    assert report.target_paper_recall is None
    assert report.evidence_recall is None
    assert report.paper_source_accuracy is None
    assert report.false_recall_rate == pytest.approx(0.5)
    assert report.false_candidate_rate == pytest.approx(0.5)
    first = report.cases[0]
    assert first.false_recall is True
    assert first.false_candidate is True
    second = report.cases[1]
    assert second.false_recall is False
    assert second.false_candidate is False


@pytest.mark.asyncio
async def test_dimension_support_needs_an_anchored_expectation() -> None:
    base_case = {
        "case_id": "compare",
        "question": "Compare results",
        "mode": "compare",
        "paper_ids": ["paper-a", "paper-b"],
        "expected_paper_ids": ["paper-a", "paper-b"],
        "sub_questions": [
            {"query": "results of paper-a", "dimension": "results", "paper_ids": ["paper-a"]},
            {"query": "results of paper-b", "dimension": "results", "paper_ids": ["paper-b"]},
        ],
    }
    anchored = RetrievalEvalDataset.model_validate(
        {
            "name": "anchored",
            "cases": [
                {
                    **base_case,
                    "expected_evidence": [
                        {"paper_id": "paper-a", "anchor": "latency", "dimension": "results"},
                        {"paper_id": "paper-b", "anchor": "throughput", "dimension": "results"},
                    ],
                }
            ],
        }
    )
    unanchored = RetrievalEvalDataset.model_validate(
        {
            "name": "unanchored",
            "cases": [{**base_case, "expected_evidence": []}],
        }
    )
    retriever = _StubRetriever(
        {
            "results of paper-a": ["paper-a|Latency improves."],
            "results of paper-b": ["paper-b|Nothing expected here."],
        },
        strategy=RetrievalStrategy.MULTI_PAPER,
    )

    report = await RetrievalEvaluator(retriever).evaluate(anchored)  # type: ignore[arg-type]
    empty = await RetrievalEvaluator(retriever).evaluate(unanchored)  # type: ignore[arg-type]

    assert report.dimension_support_rate == pytest.approx(0.5)
    assert report.cases[0].unsupported_cells == ["paper-b:results"]
    assert report.evidence_precision == pytest.approx(0.5)
    assert empty.dimension_support_rate is None
    assert empty.evidence_precision is None


def test_comparison_reports_which_case_lost_evidence() -> None:
    from app.eval.compare import compare_reports

    baseline = RetrievalEvalReport(
        dataset="baseline",
        case_count=1,
        target_paper_recall=1.0,
        evidence_recall=1.0,
        evidence_precision=1.0,
        paper_source_accuracy=1.0,
        candidate_dimension_coverage=1.0,
        dimension_support_rate=1.0,
        strategy_accuracy=1.0,
        mean_latency_ms=10.0,
        latency_p50_ms=10.0,
        latency_p95_ms=10.0,
        cases=[
            RetrievalEvalCaseResult(
                case_id="compare",
                strategy=RetrievalStrategy.MULTI_PAPER,
                strategy_matched=True,
                expects_answer=True,
                target_paper_recall=1.0,
                evidence_recall=1.0,
                hit_count=3,
                latency_ms=10.0,
            )
        ],
    )
    current = baseline.model_copy(
        update={
            "dataset": "with-distractors",
            "target_paper_recall": 0.5,
            "evidence_recall": 0.5,
            "mean_latency_ms": 14.0,
            "cases": [
                baseline.cases[0].model_copy(
                    update={
                        "target_paper_recall": 0.5,
                        "evidence_recall": 0.5,
                        "missing_paper_ids": ["paper-c"],
                        "unmatched_anchors": ["paper-c:limitations"],
                        "hit_count": 2,
                    }
                )
            ],
        }
    )

    comparison = compare_reports(baseline, current)

    assert comparison.regressed_case_ids == ["compare"]
    assert comparison.improved_case_ids == []
    case = comparison.cases[0]
    assert case.gained_missing_paper_ids == ["paper-c"]
    assert case.lost_anchors == ["paper-c:limitations"]
    assert (case.baseline_hit_count, case.current_hit_count) == (3, 2)
    metrics = {item.metric: item.delta for item in comparison.metrics}
    assert metrics["target_paper_recall"] == pytest.approx(-0.5)
    assert metrics["mean_latency_ms"] == pytest.approx(4.0)
    markdown = comparison.to_markdown()
    assert "with-distractors vs baseline" in markdown
    assert "regressed: 1" in markdown


def test_metric_definitions_and_sample_dataset_stay_in_sync() -> None:
    """Guards the documented metric vocabulary against silent renames."""

    from app.eval.compare import _METRICS

    report_fields = set(RetrievalEvalReport.model_fields)
    assert set(_METRICS) <= report_fields
    assert "hit_precision" not in report_fields
    assert {"evidence_precision", "paper_source_accuracy", "dimension_support_rate"} <= (
        report_fields
    )
