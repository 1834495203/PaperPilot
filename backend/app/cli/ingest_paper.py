import argparse
import asyncio
import re
from pathlib import Path

from app.application.paper_ingestion import PaperIngestionService
from app.cli.factories import build_embedder, build_vector_store, index_signature_for
from app.config import get_settings
from app.infrastructure.rag.pdf_parser import PypdfScientificPaperParser
from app.infrastructure.rag.tree_chunker import TreeRagChunker


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse a scientific PDF and persist its TreeRAG index in Chroma."
    )
    parser.add_argument("pdf", type=Path, help="Path to the source PDF")
    parser.add_argument("--paper-id", help="Stable paper identifier; defaults to the file name")
    parser.add_argument("--title", help="Optional title override")
    return parser.parse_args()


def _default_paper_id(path: Path) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", path.stem).strip("-_.").lower()
    return normalized or "paper"


async def _run() -> None:
    args = _arguments()
    settings = get_settings()
    signature = index_signature_for(settings)
    service = PaperIngestionService(
        parser=PypdfScientificPaperParser(),
        chunker=TreeRagChunker(
            max_chunk_chars=settings.tree_chunk_max_chars,
            overlap_sentences=settings.tree_chunk_overlap_sentences,
        ),
        embedder=build_embedder(settings),
        vector_store=build_vector_store(settings),
    )
    result = await service.ingest_pdf(
        str(args.pdf),
        paper_id=args.paper_id or _default_paper_id(args.pdf),
        title=args.title,
    )
    print(result.model_dump_json(indent=2))
    print(
        "index_signature: "
        f"{signature.fingerprint} (embedding {signature.embedding_model}, "
        f"chunker {signature.chunker_version}, parser {signature.parser_version})"
    )


if __name__ == "__main__":
    asyncio.run(_run())
