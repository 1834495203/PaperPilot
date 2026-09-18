"""Shared CLI constructors for the local TreeRAG index and retriever."""

from app.application.tree_retrieval import TreeRagRetriever
from app.config import Settings
from app.domain.rag import IndexSignature
from app.infrastructure.rag.chroma_store import ChromaTreeVectorStore
from app.infrastructure.rag.embeddings import OpenAITextEmbeddingGateway
from app.infrastructure.rag.index_signature import build_index_signature
from app.infrastructure.rag.reranker import SentenceTransformerCrossEncoderReranker


def index_signature_for(settings: Settings) -> IndexSignature:
    return build_index_signature(
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
    )


def build_embedder(settings: Settings) -> OpenAITextEmbeddingGateway:
    return OpenAITextEmbeddingGateway(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        dimensions=settings.embedding_dimensions,
    )


def build_vector_store(settings: Settings) -> ChromaTreeVectorStore:
    return ChromaTreeVectorStore(
        persist_directory=settings.vector_db_path,
        collection_name=settings.vector_collection,
        keyword_filter_limit=settings.retrieval_keyword_filter_limit,
        index_signature=index_signature_for(settings),
        enforce_index_signature=settings.vector_enforce_index_signature,
    )


def build_reranker(settings: Settings) -> SentenceTransformerCrossEncoderReranker | None:
    if not settings.reranker_model:
        return None
    return SentenceTransformerCrossEncoderReranker(
        model_name=settings.reranker_model,
        max_length=settings.reranker_max_length,
        device=settings.reranker_device,
    )


def build_retriever(settings: Settings) -> TreeRagRetriever:
    return TreeRagRetriever(
        embedder=build_embedder(settings),
        vector_store=build_vector_store(settings),
        reranker=build_reranker(settings),
        initial_top_k=settings.retrieval_initial_top_k,
        final_top_k=settings.retrieval_final_top_k,
        max_expanded_per_hit=settings.retrieval_max_expanded_per_hit,
        max_candidates=settings.retrieval_max_candidates,
        max_chunks_per_paper=settings.retrieval_max_chunks_per_paper,
        paper_top_k=settings.retrieval_paper_top_k,
        sections_per_paper=settings.retrieval_sections_per_paper,
        global_fallback_top_k=settings.retrieval_global_fallback_top_k,
        keyword_top_k=settings.retrieval_keyword_top_k,
        keyword_enabled=settings.retrieval_keyword_enabled,
        rerank_candidate_limit=settings.retrieval_rerank_candidate_limit,
        multi_paper_chunks_per_paper=settings.retrieval_multi_paper_chunks_per_paper,
        survey_paper_top_k=settings.retrieval_survey_paper_top_k,
        survey_final_top_k=settings.retrieval_survey_final_top_k,
        max_final_top_k=settings.retrieval_max_final_top_k,
        max_expansion_levels=settings.retrieval_max_expansion_levels,
        rrf_constant=settings.retrieval_rrf_constant,
        rrf_ranking_weight=settings.retrieval_rrf_ranking_weight,
        min_ranking_score=settings.retrieval_min_ranking_score,
        score_window=settings.retrieval_score_window,
        mmr_top_k=settings.retrieval_mmr_top_k,
        mmr_lambda=settings.retrieval_mmr_lambda,
    )
