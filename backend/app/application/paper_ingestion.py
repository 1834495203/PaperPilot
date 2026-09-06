from pathlib import Path

from app.domain.ports import ScientificPaperParser, TextEmbeddingGateway, TreeVectorStore
from app.domain.rag import PaperIngestionResult, TreeNodeType
from app.infrastructure.rag.tree_chunker import TreeRagChunker


class PaperIngestionService:
    def __init__(
        self,
        *,
        parser: ScientificPaperParser,
        chunker: TreeRagChunker,
        embedder: TextEmbeddingGateway,
        vector_store: TreeVectorStore,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    async def ingest_pdf(
        self,
        path: str,
        *,
        paper_id: str,
        title: str | None = None,
        asset_dir: Path | None = None,
    ) -> PaperIngestionResult:
        document = await self._parser.parse(
            path, paper_id=paper_id, title=title, asset_dir=asset_dir
        )
        nodes = self._chunker.chunk(document)
        embeddings = await self._embedder.embed_documents(
            [node.embedding_text for node in nodes]
        )
        if len(embeddings) != len(nodes):
            raise RuntimeError(
                f"Embedding provider returned {len(embeddings)} vectors for {len(nodes)} nodes"
            )
        await self._vector_store.replace_paper(document, nodes, embeddings)
        return PaperIngestionResult(
            paper_id=document.paper_id,
            metadata=document.metadata,
            page_count=document.page_count,
            section_count=len(document.sections),
            node_count=len(nodes),
            chunk_count=sum(node.node_type is TreeNodeType.CHUNK for node in nodes),
            vector_collection=self._vector_store.collection_name,
        )
