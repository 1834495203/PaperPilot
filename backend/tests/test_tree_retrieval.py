from collections.abc import Sequence

import pytest

from app.application.tree_retrieval import TreeRagRetriever
from app.domain.ports import TextEmbeddingGateway, TreeVectorStore
from app.domain.rag import (
    IndexedTreeNode,
    ParsedPaperDocument,
    RetrievalMode,
    RetrievalSource,
    TreeIndexNode,
    TreeNodeType,
    TreeVectorMatch,
)


def _node(
    node_id: str,
    node_type: TreeNodeType,
    *,
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    text: str = "",
) -> TreeIndexNode:
    return TreeIndexNode(
        node_id=node_id,
        paper_id="paper",
        node_type=node_type,
        title=node_id,
        parent_id=parent_id,
        children_ids=children_ids or [],
        level=1,
        section_path=["3 Method"],
        text=text,
        embedding_text=text or node_id,
        page_start=3,
        page_end=3,
    )


class _Embedder(TextEmbeddingGateway):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _Store(TreeVectorStore):
    def __init__(self) -> None:
        self.chunk_a = _node(
            "chunk-a",
            TreeNodeType.CHUNK,
            parent_id="method",
            text="TreeRAG builds a hierarchical index.",
        )
        self.chunk_b = _node(
            "chunk-b",
            TreeNodeType.CHUNK,
            parent_id="method",
            text="Sibling nodes restore related context.",
        )
        self.method = _node(
            "method",
            TreeNodeType.SECTION,
            parent_id="root",
            children_ids=["chunk-a", "chunk-b"],
        )
        self.root = _node(
            "root",
            TreeNodeType.ROOT,
            children_ids=["method"],
        )
        self.last_chunks_only: bool | None = None
        self.load_count = 0

    @property
    def collection_name(self) -> str:
        return "test_tree"

    async def replace_paper(
        self,
        document: ParsedPaperDocument,
        nodes: Sequence[TreeIndexNode],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        return None

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str],
        top_k: int,
        chunks_only: bool,
    ) -> list[TreeVectorMatch]:
        self.last_chunks_only = chunks_only
        return [TreeVectorMatch(node=self.chunk_a, vector_score=1.0)]

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        self.load_count += 1
        return [
            IndexedTreeNode(node=self.root, embedding=[0.1, 0.9]),
            IndexedTreeNode(node=self.method, embedding=[0.8, 0.2]),
            IndexedTreeNode(node=self.chunk_a, embedding=[1.0, 0.0]),
            IndexedTreeNode(node=self.chunk_b, embedding=[0.9, 0.1]),
        ]


@pytest.mark.asyncio
async def test_fact_mode_returns_direct_chunk_hits_without_tree_expansion() -> None:
    store = _Store()
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=store)

    report = await retriever.retrieve(
        "How is the index built?",
        paper_ids=["paper"],
        mode=RetrievalMode.FACT,
    )

    assert store.last_chunks_only is True
    assert store.load_count == 0
    assert [hit.node_id for hit in report.hits] == ["chunk-a"]
    assert report.hits[0].source is RetrievalSource.VECTOR


@pytest.mark.asyncio
async def test_method_mode_expands_a_leaf_to_related_sibling_chunks() -> None:
    store = _Store()
    retriever = TreeRagRetriever(embedder=_Embedder(), vector_store=store)

    report = await retriever.retrieve(
        "Explain the retrieval method",
        paper_ids=["paper"],
        mode=RetrievalMode.METHOD,
    )

    assert store.last_chunks_only is False
    assert store.load_count == 1
    assert [hit.node_id for hit in report.hits] == ["chunk-a", "chunk-b"]
    sibling = report.hits[1]
    assert sibling.source is RetrievalSource.TREE_EXPANSION
    assert sibling.expanded_from == "chunk-a"
    assert report.expanded_candidate_count == 1
    assert sibling.ranking_score != sibling.vector_score
