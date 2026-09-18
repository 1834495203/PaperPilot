import argparse
import asyncio
import sys
from pathlib import Path

from app.cli.factories import build_vector_store, index_signature_for
from app.config import get_settings
from app.domain.rag import IndexedPaper, PaperIndexStatus


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report which indexed papers were built by the currently configured "
            "embedding model, parser and chunker."
        )
    )
    parser.add_argument(
        "--library-path",
        type=Path,
        default=None,
        help="Override the configured paper library directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the status report as JSON",
    )
    return parser.parse_args()


def _manifests(library_path: Path) -> list[IndexedPaper]:
    records: list[IndexedPaper] = []
    for path in sorted(library_path.glob("*.json")):
        try:
            records.append(IndexedPaper.model_validate_json(path.read_text("utf-8")))
        except (OSError, ValueError):
            continue
    return records


async def _run() -> None:
    args = _arguments()
    settings = get_settings()
    library_path = args.library_path or settings.paper_library_path
    signature = index_signature_for(settings)
    store = build_vector_store(settings)
    statuses = [
        PaperIndexStatus(
            paper_id=paper.paper_id,
            title=paper.title,
            chunk_count=paper.chunk_count,
            index_signature=paper.index_signature,
            current_signature=signature.fingerprint,
        )
        for paper in _manifests(library_path)
    ]
    rebuild = [item for item in statuses if item.needs_rebuild]
    if args.as_json:
        payload = (
            "["
            + ",".join(item.model_dump_json() for item in statuses)
            + "]"
        )
        sys.stdout.buffer.write(payload.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")
        return
    print(
        f"collection={store.collection_name} "
        f"stored_index_signature={store.stored_index_signature or 'unknown'} "
        f"current_index_signature={signature.fingerprint} "
        f"signature_matches={store.index_signature_matches}"
    )
    for item in statuses:
        state = (
            "REBUILD"
            if item.needs_rebuild
            else ("ok" if item.fingerprint_recorded else "unknown-fingerprint")
        )
        print(
            f"{state:>20}  {item.paper_id}  chunks={item.chunk_count}  "
            f"signature={item.index_signature or 'not-recorded'}  {item.title}"
        )
    print(f"papers={len(statuses)} needing_rebuild={len(rebuild)}")


if __name__ == "__main__":
    asyncio.run(_run())
