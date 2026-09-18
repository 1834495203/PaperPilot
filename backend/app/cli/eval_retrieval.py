import argparse
import asyncio
import sys
from pathlib import Path

from app.cli.factories import build_retriever
from app.config import get_settings
from app.eval.compare import compare_reports
from app.eval.dataset import RetrievalEvalReport
from app.eval.runner import RetrievalEvaluator, load_dataset


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TreeRAG retrieval against a labelled dataset. Metrics cover whether "
            "expected papers and evidence were missed (target-paper recall, evidence recall), "
            "whether what returned is the right evidence (evidence precision, paper source "
            "accuracy, dimension support), unanswerable-question false recalls, and latency. "
            "Use --baseline to diff two runs, for example before and after adding distractor "
            "papers."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help=(
            "JSON dataset of labelled cases; see "
            "tests/fixtures/retrieval_eval.sample.json for the format"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the full JSON report to this path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Write a Markdown summary to this path",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Earlier JSON report to diff against, for example the same dataset run "
            "before distractor papers were added to the library"
        ),
    )
    return parser.parse_args()


async def _run() -> None:
    args = _arguments()
    settings = get_settings()
    dataset = load_dataset(args.dataset.read_text(encoding="utf-8"))
    retriever = build_retriever(settings)
    report = await RetrievalEvaluator(retriever).evaluate(dataset)
    if args.output is not None:
        args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(report.to_markdown(), encoding="utf-8")
    sys.stdout.buffer.write(report.to_markdown().encode("utf-8"))
    if args.baseline is not None:
        baseline = RetrievalEvalReport.model_validate_json(
            args.baseline.read_text(encoding="utf-8")
        )
        comparison = compare_reports(baseline, report)
        comparison_path = args.output.with_name(
            f"{args.output.stem}.comparison.json"
        ) if args.output is not None else None
        if comparison_path is not None:
            comparison_path.write_text(
                comparison.model_dump_json(indent=2), encoding="utf-8"
            )
        sys.stdout.buffer.write(comparison.to_markdown().encode("utf-8"))


if __name__ == "__main__":
    asyncio.run(_run())
