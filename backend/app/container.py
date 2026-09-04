from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import create_async_engine

from app.application.chat_service import ChatService
from app.application.paper_ingestion import PaperIngestionService
from app.application.paper_library import PaperLibraryService
from app.application.tree_retrieval import TreeRagRetriever
from app.config import Settings
from app.domain.papers import PaperSource
from app.infrastructure.agent.supervisor.factory import create_supervisor_agent
from app.infrastructure.db.store import SqlAlchemyConversationStore
from app.infrastructure.rag.chroma_store import ChromaTreeVectorStore
from app.infrastructure.rag.embeddings import OpenAITextEmbeddingGateway
from app.infrastructure.rag.pdf_parser import PypdfScientificPaperParser
from app.infrastructure.rag.tree_chunker import TreeRagChunker
from app.infrastructure.tools.arxiv import ArxivPaperSearchGateway
from app.infrastructure.tools.openalex import OpenAlexPaperSearchGateway
from app.infrastructure.tools.paper_search import (
    FallbackPaperSearchGateway,
    SearchProvider,
)
from app.infrastructure.tools.pdf import ArxivPdfDocumentGateway
from app.infrastructure.tools.semantic_scholar import SemanticScholarPaperSearchGateway


class ApplicationContainer:
    def __init__(self, settings: Settings) -> None:
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        self.store = SqlAlchemyConversationStore(engine)
        paper_search = FallbackPaperSearchGateway(
            [
                SearchProvider(
                    PaperSource.OPENALEX,
                    OpenAlexPaperSearchGateway(
                        api_url=settings.openalex_api_url,
                        api_key=settings.openalex_api_key.get_secret_value(),
                        timeout_seconds=settings.openalex_timeout_seconds,
                        min_request_interval_seconds=(
                            settings.openalex_min_request_interval_seconds
                        ),
                        max_retries=settings.openalex_max_retries,
                        retry_backoff_seconds=settings.openalex_retry_backoff_seconds,
                    ),
                ),
                SearchProvider(
                    PaperSource.SEMANTIC_SCHOLAR,
                    SemanticScholarPaperSearchGateway(
                        api_url=settings.semantic_scholar_api_url,
                        api_key=settings.semantic_scholar_api_key.get_secret_value(),
                        timeout_seconds=settings.semantic_scholar_timeout_seconds,
                        min_request_interval_seconds=(
                            settings.semantic_scholar_min_request_interval_seconds
                        ),
                        max_retries=settings.semantic_scholar_max_retries,
                        retry_backoff_seconds=(settings.semantic_scholar_retry_backoff_seconds),
                    ),
                ),
                SearchProvider(
                    PaperSource.ARXIV,
                    ArxivPaperSearchGateway(
                        api_url=settings.arxiv_api_url,
                        timeout_seconds=settings.arxiv_timeout_seconds,
                        min_request_interval_seconds=(settings.arxiv_min_request_interval_seconds),
                        max_retries=settings.arxiv_max_retries,
                        retry_backoff_seconds=settings.arxiv_retry_backoff_seconds,
                    ),
                ),
            ]
        )
        paper_document = ArxivPdfDocumentGateway(
            timeout_seconds=settings.pdf_timeout_seconds,
            max_bytes=settings.pdf_max_bytes,
            max_pages=settings.pdf_max_pages,
            max_characters=settings.reader_max_input_chars,
        )
        embedder = OpenAITextEmbeddingGateway(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            dimensions=settings.embedding_dimensions,
        )
        vector_store = ChromaTreeVectorStore(
            persist_directory=settings.vector_db_path,
            collection_name=settings.vector_collection,
        )
        retriever = TreeRagRetriever(
            embedder=embedder,
            vector_store=vector_store,
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
        )
        paper_parser = PypdfScientificPaperParser()
        ingestion = PaperIngestionService(
            parser=paper_parser,
            chunker=TreeRagChunker(max_chunk_chars=settings.tree_chunk_max_chars),
            embedder=embedder,
            vector_store=vector_store,
        )
        self.paper_library = PaperLibraryService(
            library_path=settings.paper_library_path,
            max_upload_bytes=settings.upload_max_bytes,
            ingestion=ingestion,
            retriever=retriever,
            vector_store=vector_store,
            metadata_parser=paper_parser,
        )
        api_key = settings.llm_api_key
        if not api_key.get_secret_value():
            api_key = SecretStr("not-configured")
        model = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=api_key,
            base_url=settings.llm_base_url,
            stream_usage=True,
        )
        agent = create_supervisor_agent(
            model=model,
            paper_search=paper_search,
            paper_document=paper_document,
            store=self.store,
            max_steps=settings.supervisor_max_steps,
            search_max_iterations=settings.search_max_iterations,
            reader_max_retrieval_rounds=settings.reader_max_retrieval_rounds,
            paper_retriever=retriever,
            paper_library=self.paper_library,
        )
        self.chat_service = ChatService(self.store, agent)

    async def initialize(self) -> None:
        await self.store.initialize()
        await self.paper_library.initialize()

    async def close(self) -> None:
        await self.store.close()
