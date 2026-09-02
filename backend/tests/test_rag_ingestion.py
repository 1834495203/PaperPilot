from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from app.application.paper_ingestion import PaperIngestionService
from app.domain.ports import ScientificPaperParser, TextEmbeddingGateway, TreeVectorStore
from app.domain.rag import (
    IndexedTreeNode,
    PageTextBlock,
    PaperMetadata,
    PaperSection,
    ParsedPaperDocument,
    TreeIndexNode,
    TreeNodeType,
    TreeVectorMatch,
)
from app.infrastructure.rag.chroma_store import ChromaTreeVectorStore
from app.infrastructure.rag.pdf_parser import PypdfScientificPaperParser
from app.infrastructure.rag.tree_chunker import TreeRagChunker


def sample_document(source_path: Path) -> ParsedPaperDocument:
    method = PaperSection(
        section_id="paper:section:0001",
        index="3",
        title="Method",
        level=1,
        blocks=[],
        page_start=3,
        page_end=3,
    )
    retrieval = PaperSection(
        section_id="paper:section:0002",
        index="3.1",
        title="Retrieval",
        level=2,
        parent_section_id=method.section_id,
        blocks=[
            PageTextBlock(
                page_number=4,
                text=(
                    "The system retrieves semantically related passages.\n\n"
                    "It then expands sibling nodes to improve recall."
                ),
            )
        ],
        page_start=4,
        page_end=4,
    )
    return ParsedPaperDocument(
        paper_id="paper",
        metadata=PaperMetadata(title="Tree Structured Retrieval"),
        source_path=source_path,
        page_count=5,
        sections=[method, retrieval],
    )


@pytest.mark.asyncio
async def test_parser_recovers_treerag_native_section_hierarchy() -> None:
    source = Path(__file__).parents[1] / "paper" / "TreeRAG.pdf"

    document = await PypdfScientificPaperParser().parse(
        str(source),
        paper_id="treerag",
    )

    by_index = {section.index: section for section in document.sections if section.index}
    assert document.title.startswith("TreeRAG: Unleashing the Power")
    assert document.metadata.authors[:2] == ["Wenyu Tao", "Xiaofen Xing"]
    assert document.metadata.abstract is not None
    assert document.metadata.abstract.startswith("When confronting long document")
    assert document.page_count == 16
    assert by_index["3.1"].parent_section_id == by_index["3"].section_id
    assert by_index["3.1.1"].parent_section_id == by_index["3.1"].section_id
    assert by_index["A.5"].parent_section_id == by_index["A"].section_id


def test_tree_chunker_prefixes_content_with_all_ancestor_titles(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")

    nodes = TreeRagChunker(max_chunk_chars=200).chunk(document)

    by_id = {node.node_id: node for node in nodes}
    retrieval = by_id["paper:section:0002"]
    chunks = [node for node in nodes if node.node_type is TreeNodeType.CHUNK]
    assert retrieval.parent_id == "paper:section:0001"
    assert retrieval.node_id in by_id["paper:section:0001"].children_ids
    assert chunks
    assert chunks[0].section_path == ["3 Method", "3.1 Retrieval"]
    assert chunks[0].embedding_text.startswith(
        "Tree Structured Retrieval\n3 Method\n3.1 Retrieval\n"
    )
    assert chunks[0].text.startswith("The system retrieves")
    assert chunks[0].page_start == 4


class _FakeParser(ScientificPaperParser):
    def __init__(self, document: ParsedPaperDocument) -> None:
        self.document = document

    async def parse(
        self,
        path: str,
        *,
        paper_id: str,
        title: str | None = None,
    ) -> ParsedPaperDocument:
        return self.document


class _FakeEmbedder(TextEmbeddingGateway):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 1.0]


class _FakeStore(TreeVectorStore):
    def __init__(self) -> None:
        self.saved_nodes: list[TreeIndexNode] = []
        self.saved_embeddings: list[list[float]] = []

    @property
    def collection_name(self) -> str:
        return "test_tree"

    async def replace_paper(
        self,
        document: ParsedPaperDocument,
        nodes: Sequence[TreeIndexNode],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.saved_nodes = list(nodes)
        self.saved_embeddings = [list(item) for item in embeddings]

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str],
        top_k: int,
        chunks_only: bool,
    ) -> list[TreeVectorMatch]:
        return []

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        return []


@pytest.mark.asyncio
async def test_ingestion_service_embeds_and_persists_every_tree_node(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")
    store = _FakeStore()
    service = PaperIngestionService(
        parser=_FakeParser(document),
        chunker=TreeRagChunker(max_chunk_chars=200),
        embedder=_FakeEmbedder(),
        vector_store=store,
    )

    result = await service.ingest_pdf("ignored.pdf", paper_id="paper")

    assert result.node_count == len(store.saved_nodes)
    assert result.chunk_count >= 1
    assert len(store.saved_embeddings) == len(store.saved_nodes)
    assert result.vector_collection == "test_tree"


@pytest.mark.asyncio
async def test_chroma_store_persists_tree_edges_and_original_text(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    document = sample_document(tmp_path / "paper.pdf")
    nodes = TreeRagChunker(max_chunk_chars=200).chunk(document)
    embeddings = [
        [float(index + 1), 1.0, 0.5]
        for index, _ in enumerate(nodes)
    ]
    store = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_tree_chunks",
    )

    await store.replace_paper(document, nodes, embeddings)

    collection = cast(object, store)._collection  # type: ignore[attr-defined]
    result = collection.get(where={"paper_id": "paper"})  # type: ignore[attr-defined]
    assert len(result["ids"]) == len(nodes)
    chunk_index = next(
        index
        for index, metadata in enumerate(result["metadatas"])
        if metadata["node_type"] == "chunk"
    )
    assert result["documents"][chunk_index].startswith("The system retrieves")
    assert result["metadatas"][chunk_index]["parent_id"] == "paper:section:0002"
    assert "3 Method" in result["metadatas"][chunk_index]["section_path"]

    matches = await store.similarity_search(
        embeddings[chunk_index],
        paper_ids=["paper"],
        top_k=3,
        chunks_only=True,
    )
    indexed_nodes = await store.load_paper_nodes(["paper"])
    assert matches
    assert all(match.node.node_type is TreeNodeType.CHUNK for match in matches)
    assert len(indexed_nodes) == len(nodes)
    assert all(len(item.embedding) == 3 for item in indexed_nodes)
