import argparse
import asyncio
import sys

from app.cli.factories import build_retriever
from app.config import get_settings
from app.domain.rag import RetrievalMode, RetrievalQuery, RetrievalStrategy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve TreeRAG chunks from the local Chroma paper index using one query "
            "or an explicit sub-question plan."
        )
    )
    parser.add_argument("query", help="Question or retrieval query")
    parser.add_argument(
        "--paper-id",
        action="append",
        required=True,
        dest="paper_ids",
        help="Paper identifier; repeat the option to search multiple papers",
    )
    parser.add_argument(
        "--mode",
        choices=[item.value for item in RetrievalMode],
        default=RetrievalMode.FACT.value,
    )
    parser.add_argument(
        "--strategy",
        choices=[item.value for item in RetrievalStrategy],
        default=None,
        help="Retrieval task strategy; omitted means the retriever infers it",
    )
    parser.add_argument(
        "--sub-question",
        action="append",
        dest="sub_questions",
        default=[],
        metavar="DIMENSION=QUERY",
        help=(
            "Additional sub-question as DIMENSION=QUERY; repeat once per paper x "
            "dimension to retrieve a multi-paper comparison"
        ),
    )
    parser.add_argument(
        "--sub-mode",
        choices=[item.value for item in RetrievalMode],
        default=RetrievalMode.METHOD.value,
        help="Retrieval mode applied to every --sub-question",
    )
    parser.add_argument(
        "--global",
        action="store_true",
        dest="search_globally",
        help="Ignore --paper-id and search the whole indexed corpus",
    )
    return parser.parse_args()


def _sub_queries(args: argparse.Namespace) -> list[RetrievalQuery] | None:
    if not args.sub_questions:
        return None
    queries: list[RetrievalQuery] = []
    for item in args.sub_questions:
        dimension, _, text = str(item).partition("=")
        if not text:
            dimension, text = "", dimension
        queries.append(
            RetrievalQuery(
                query=text.strip(),
                mode=RetrievalMode(str(args.sub_mode)),
                paper_ids=[] if args.search_globally else list(args.paper_ids),
                dimension=dimension.strip() or None,
            )
        )
    return queries


async def _run() -> None:
    args = _arguments()
    settings = get_settings()
    retriever = build_retriever(settings)
    strategy = None if args.strategy is None else RetrievalStrategy(str(args.strategy))
    report = await retriever.retrieve(
        args.query,
        paper_ids=None if args.search_globally else list(args.paper_ids),
        mode=RetrievalMode(str(args.mode)),
        strategy=strategy,
        queries=_sub_queries(args),
    )
    payload = report.model_dump_json(indent=2)
    sys.stdout.buffer.write(payload.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    asyncio.run(_run())
