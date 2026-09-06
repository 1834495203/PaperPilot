import argparse
import asyncio
import sys

from app.application.tree_retrieval import TreeRagRetriever
from app.config import get_settings
from app.domain.rag import RetrievalMode
from app.infrastructure.rag.chroma_store import ChromaTreeVectorStore
from app.infrastructure.rag.embeddings import OpenAITextEmbeddingGateway
from app.infrastructure.rag.reranker import SentenceTransformerCrossEncoderReranker


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve TreeRAG chunks from the local Chroma paper index."
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
    return parser.parse_args()


async def _run() -> None:
    args = _arguments()
    settings = get_settings()
    reranker = (
        SentenceTransformerCrossEncoderReranker(
            model_name=settings.reranker_model,
            max_length=settings.reranker_max_length,
            device=settings.reranker_device,
        )
        if settings.reranker_model
        else None
    )
    retriever = TreeRagRetriever(
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
        reranker=reranker,
        initial_top_k=settings.retrieval_initial_top_k,
        final_top_k=settings.retrieval_final_top_k,
        max_expanded_per_hit=settings.retrieval_max_expanded_per_hit,
        max_candidates=settings.retrieval_max_candidates,
        max_chunks_per_paper=settings.retrieval_max_chunks_per_paper,
        paper_top_k=settings.retrieval_paper_top_k,
        sections_per_paper=settings.retrieval_sections_per_paper,
        global_fallback_top_k=settings.retrieval_global_fallback_top_k,
        min_ranking_score=settings.retrieval_min_ranking_score,
        score_window=settings.retrieval_score_window,
        mmr_top_k=settings.retrieval_mmr_top_k,
        mmr_lambda=settings.retrieval_mmr_lambda,
    )
    report = await retriever.retrieve(
        args.query,
        paper_ids=args.paper_ids,
        mode=RetrievalMode(args.mode),
    )
    payload = report.model_dump_json(indent=2)
    sys.stdout.buffer.write(payload.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    asyncio.run(_run())
