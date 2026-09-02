import argparse
import asyncio
import re
from pathlib import Path

from app.application.paper_ingestion import PaperIngestionService
from app.config import get_settings
from app.infrastructure.rag.chroma_store import ChromaTreeVectorStore
from app.infrastructure.rag.embeddings import OpenAITextEmbeddingGateway
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
    service = PaperIngestionService(
        parser=PypdfScientificPaperParser(),
        chunker=TreeRagChunker(max_chunk_chars=settings.tree_chunk_max_chars),
        embedder=OpenAITextEmbeddingGateway(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            dimensions=settings.embedding_dimensions,
        ),
        vector_store=ChromaTreeVectorStore(
            persist_directory=settings.vector_db_path,
            collection_name=settings.vector_collection,
        ),
    )
    result = await service.ingest_pdf(
        str(args.pdf),
        paper_id=args.paper_id or _default_paper_id(args.pdf),
        title=args.title,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(_run())
