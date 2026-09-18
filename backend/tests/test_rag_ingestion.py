import types
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from app.application.paper_ingestion import PaperIngestionService
from app.domain.ports import ScientificPaperParser, TextEmbeddingGateway, TreeVectorStore
from app.domain.rag import (
    BoundingBox,
    CaptionBlock,
    CodeBlock,
    FigureBlock,
    IndexedTreeNode,
    KeywordSearchResult,
    PageTextBlock,
    PaperBlockType,
    PaperMetadata,
    PaperSection,
    ParsedPaperDocument,
    TableBlock,
    TreeIndexNode,
    TreeNodeType,
    TreeVectorMatch,
)
from app.infrastructure.rag.chroma_store import ChromaTreeVectorStore
from app.infrastructure.rag.pdf_parser import PypdfScientificPaperParser, _LayoutLine
from app.infrastructure.rag.tree_chunker import TreeRagChunker

_LONG_TEST_SENTENCES = [
    (
        "The first sentence contains enough additional words to exceed the minimum chunk "
        "budget on its own without splitting."
    ),
    (
        "The second sentence is similarly made long enough that a pair of these sentences "
        "cannot fit together in one chunk."
    ),
    (
        "The third sentence continues the same pattern and should land within its own "
        "separate chunk for this assertion."
    ),
    (
        "The fourth sentence ends the paragraph and confirms the boundary behaviour across "
        "the entire chunking run."
    ),
]


def test_chroma_filter_combines_tree_level_and_parent_scope() -> None:
    where = ChromaTreeVectorStore._where_filter(  # noqa: SLF001
        ["paper-a", "paper-b"],
        chunks_only=False,
        node_types=[TreeNodeType.SECTION],
        parent_ids=["root-a", "root-b"],
    )

    assert where == {
        "$and": [
            {"paper_id": {"$in": ["paper-a", "paper-b"]}},
            {"node_type": TreeNodeType.SECTION.value},
            {"parent_id": {"$in": ["root-a", "root-b"]}},
        ]
    }


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


def test_layout_style_recognizes_a_heading_without_a_template_name() -> None:
    parser = PypdfScientificPaperParser()
    heading = _LayoutLine(
        page_number=2,
        page_width=600,
        page_height=800,
        text="Failure Analysis Under Domain Shift",
        bbox=BoundingBox(x0=70, y0=120, x1=290, y1=136),
        spans=(),
        font_size=12,
        bold=True,
        italic=False,
        gap_before=10,
    )

    assert parser._is_heading_candidate(  # noqa: SLF001
        heading,
        body_font_size=10,
        style_counts=Counter({heading.style_key: 3}),
    )


def test_heading_candidate_rejects_small_font_figure_labels() -> None:
    parser = PypdfScientificPaperParser()
    label = _LayoutLine(
        page_number=2,
        page_width=600,
        page_height=800,
        text="Step 1: Retrieve K documents",
        bbox=BoundingBox(x0=70, y0=120, x1=220, y1=132),
        spans=(),
        font_size=5.2,
        bold=True,
        italic=False,
    )

    assert not parser._is_heading_candidate(  # noqa: SLF001
        label,
        body_font_size=10,
        style_counts=Counter({label.style_key: 3}),
    )


def test_looks_like_heading_title_allows_question_marks() -> None:
    parser = PypdfScientificPaperParser()

    assert parser._looks_like_heading_title(  # noqa: SLF001
        "How Well Can Language Models Retrieve From Input Contexts?"
    )
    assert not parser._looks_like_heading_title("This is a normal sentence.")  # noqa: SLF001


def test_sentence_continues_across_page_break() -> None:
    parser = PypdfScientificPaperParser()

    assert parser._sentence_continues(  # noqa: SLF001
        "clustering issues due", "to the lack of clear subject."
    )
    assert not parser._sentence_continues(  # noqa: SLF001
        "This is a complete sentence.", "Next sentence starts here."
    )
    assert not parser._sentence_continues(  # noqa: SLF001
        "Ends here.", "lowercase fragment"
    )
    assert not parser._sentence_continues(  # noqa: SLF001
        "ends without period", "Uppercase start"
    )


def test_extract_code_blocks_detects_boxed_algorithm() -> None:
    parser = PypdfScientificPaperParser()
    page = types.SimpleNamespace(
        rect=types.SimpleNamespace(width=600.0, height=800.0),
        get_drawings=lambda: [
            {
                "rect": types.SimpleNamespace(
                    height=0.0, width=250.0, y0=100.0, x0=70.0, x1=320.0
                )
            },
            {
                "rect": types.SimpleNamespace(
                    height=0.0, width=250.0, y0=130.0, x0=70.0, x1=320.0
                )
            },
            {
                "rect": types.SimpleNamespace(
                    height=0.0, width=250.0, y0=300.0, x0=70.0, x1=320.0
                )
            },
        ],
    )
    lines = [
        _LayoutLine(
            page_number=1,
            page_width=600,
            page_height=800,
            text="Algorithm 1: Foo",
            bbox=BoundingBox(x0=75, y0=105, x1=200, y1=120),
            spans=(),
            font_size=10,
            bold=True,
            italic=False,
        ),
        _LayoutLine(
            page_number=1,
            page_width=600,
            page_height=800,
            text="1 x = 1",
            bbox=BoundingBox(x0=75, y0=140, x1=150, y1=155),
            spans=(),
            font_size=10,
            bold=False,
            italic=False,
        ),
        _LayoutLine(
            page_number=1,
            page_width=600,
            page_height=800,
            text="body text",
            bbox=BoundingBox(x0=75, y0=320, x1=150, y1=335),
            spans=(),
            font_size=10,
            bold=False,
            italic=False,
        ),
    ]

    code_blocks, remaining = parser._extract_code_blocks(page, 1, lines)  # noqa: SLF001

    assert len(code_blocks) == 1
    assert "Algorithm 1: Foo" in code_blocks[0].text
    assert "1 x = 1" in code_blocks[0].text
    assert [line.text for line in remaining] == ["body text"]


def test_merge_caption_continuations_keeps_caption_intact() -> None:
    parser = PypdfScientificPaperParser()
    caption = _LayoutLine(
        page_number=4,
        page_width=600,
        page_height=800,
        text="Figure 4: Unsatisfactory Vector",
        bbox=BoundingBox(x0=70, y0=100, x1=250, y1=112),
        spans=(),
        font_size=10,
        bold=False,
        italic=False,
    )
    continuation = _LayoutLine(
        page_number=4,
        page_width=600,
        page_height=800,
        text="Distance.",
        bbox=BoundingBox(x0=70, y0=112, x1=120, y1=124),
        spans=(),
        font_size=10,
        bold=False,
        italic=False,
    )
    body = _LayoutLine(
        page_number=4,
        page_width=600,
        page_height=800,
        text="This is body text.",
        bbox=BoundingBox(x0=70, y0=200, x1=160, y1=212),
        spans=(),
        font_size=10,
        bold=False,
        italic=False,
    )

    merged = parser._merge_caption_continuations(  # noqa: SLF001
        [caption, continuation, body]
    )

    assert [line.text for line in merged] == [
        "Figure 4: Unsatisfactory Vector Distance.",
        "This is body text.",
    ]


def test_parser_recognizes_page_number_lines_only_near_edges() -> None:
    parser = PypdfScientificPaperParser()
    page_number = _LayoutLine(
        page_number=3,
        page_width=600,
        page_height=800,
        text="3",
        bbox=BoundingBox(x0=290, y0=770, x1=310, y1=790),
        spans=(),
        font_size=9,
        bold=False,
        italic=False,
    )
    body_number = _LayoutLine(
        page_number=3,
        page_width=600,
        page_height=800,
        text="80",
        bbox=BoundingBox(x0=70, y0=400, x1=120, y1=415),
        spans=(),
        font_size=10,
        bold=False,
        italic=False,
    )

    assert parser._is_page_number_line(page_number)  # noqa: SLF001
    assert not parser._is_page_number_line(body_number)  # noqa: SLF001


@pytest.mark.asyncio
async def test_parser_recovers_borderless_tables_and_paperqa_subsections() -> None:
    source = Path(__file__).parents[1] / "paper" / "PaperQA.pdf"

    document = await PypdfScientificPaperParser().parse(
        str(source),
        paper_id="paperqa",
    )

    by_index = {section.index: section for section in document.sections if section.index}
    table_blocks = [
        block
        for section in document.sections
        for block in section.blocks
        if isinstance(block, TableBlock)
    ]
    assert by_index["3.1"].title == "PAPERQA"
    assert by_index["3.1"].parent_section_id == by_index["3"].section_id
    assert table_blocks
    assert all(block.caption for block in table_blocks)


def test_tree_chunker_prefixes_content_with_all_ancestor_titles(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")
    document = document.model_copy(
        update={
            "metadata": document.metadata.model_copy(
                update={
                    "abstract": "A hierarchy improves scientific retrieval.",
                    "keywords": ["retrieval", "RAG"],
                }
            )
        }
    )

    nodes = TreeRagChunker(max_chunk_chars=200).chunk(document)

    by_id = {node.node_id: node for node in nodes}
    retrieval = by_id["paper:section:0002"]
    root = by_id["paper:root"]
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
    assert "Abstract: A hierarchy improves" in root.embedding_text
    assert "Keywords: retrieval, RAG" in root.embedding_text
    assert "Sections: 3 Method; 3.1 Retrieval" in root.embedding_text
    assert "The system retrieves semantically" in retrieval.embedding_text


def test_tree_chunker_keeps_layout_objects_atomic_and_searchable(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")
    section = document.sections[-1]
    table = TableBlock(
        page_number=4,
        text=(
            "Table 2: Accuracy\n| Model | Score |\n| --- | --- |\n| Ours | 91 |"
            + " supporting evidence" * 12
        ),
        caption="Table 2: Accuracy",
        object_label="2",
        rows=[["Model", "Score"], ["Ours", "91"]],
    )
    document = document.model_copy(
        update={
            "sections": [
                *document.sections[:-1],
                section.model_copy(update={"blocks": [*section.blocks, table]}),
            ]
        }
    )

    chunks = [
        node
        for node in TreeRagChunker(max_chunk_chars=200).chunk(document)
        if node.node_type is TreeNodeType.CHUNK
    ]
    table_chunk = next(node for node in chunks if PaperBlockType.TABLE in node.block_types)
    assert table_chunk.object_labels == ["2"]
    assert table_chunk.text.startswith("Table 2: Accuracy")
    assert table_chunk.table_rows == [["Model", "Score"], ["Ours", "91"]]
    assert len(table_chunk.text) > 200


def test_tree_chunker_preserves_figure_asset_and_table_rows(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")
    section = document.sections[-1]
    figure = FigureBlock(
        page_number=4,
        text="Figure 1: Overview",
        caption="Figure 1: Overview",
        object_label="1",
        asset_ref="fig-0004-00.png",
        raw_asset_ref="raw-0004-253.png",
    )
    table = TableBlock(
        page_number=4,
        text="Table 1: Results\n| A | B |\n| 1 | 2 |",
        caption="Table 1: Results",
        object_label="1",
        rows=[["A", "B"], ["1", "2"]],
    )
    document = document.model_copy(
        update={
            "sections": [
                *document.sections[:-1],
                section.model_copy(update={"blocks": [*section.blocks, figure, table]}),
            ]
        }
    )

    chunks = [
        node
        for node in TreeRagChunker(max_chunk_chars=200).chunk(document)
        if node.node_type is TreeNodeType.CHUNK
    ]
    figure_chunk = next(node for node in chunks if PaperBlockType.FIGURE in node.block_types)
    table_chunk = next(node for node in chunks if PaperBlockType.TABLE in node.block_types)
    assert figure_chunk.figure_asset == "fig-0004-00.png"
    assert figure_chunk.figure_caption == "Figure 1: Overview"
    assert figure_chunk.raw_asset_ref == "raw-0004-253.png"
    assert table_chunk.table_rows == [["A", "B"], ["1", "2"]]


def test_tree_chunker_produces_normalized_spans(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")
    section = document.sections[-1]
    block = PageTextBlock(
        page_number=4,
        text="The system retrieves semantically related passages.",
        bbox=BoundingBox(x0=100, y0=200, x1=300, y1=220),
    )
    document = document.model_copy(
        update={
            "page_dimensions": {4: (400.0, 800.0)},
            "sections": [
                *document.sections[:-1],
                section.model_copy(update={"blocks": [block]}),
            ],
        }
    )

    chunks = [
        node
        for node in TreeRagChunker(max_chunk_chars=200).chunk(document)
        if node.node_type is TreeNodeType.CHUNK
    ]
    assert chunks and chunks[0].spans
    span = chunks[0].spans[0]
    assert span.page_number == 4
    assert span.bbox.x0 == pytest.approx(0.25)
    assert span.bbox.y0 == pytest.approx(0.25)
    assert span.bbox.x1 == pytest.approx(0.75)


def _document_with_text(tmp_path: Path, text: str) -> ParsedPaperDocument:
    document = sample_document(tmp_path / "paper.pdf")
    section = document.sections[-1]
    block = PageTextBlock(page_number=4, text=text)
    return document.model_copy(
        update={
            "sections": [
                *document.sections[:-1],
                section.model_copy(update={"blocks": [block]}),
            ]
        }
    )


def test_tree_chunker_splits_chunks_on_sentence_boundaries(tmp_path: Path) -> None:
    sentences = _LONG_TEST_SENTENCES
    document = _document_with_text(tmp_path, " ".join(sentences))

    chunks = [
        node.text
        for node in TreeRagChunker(max_chunk_chars=200, overlap_sentences=0).chunk(document)
        if node.node_type is TreeNodeType.CHUNK
    ]

    assert chunks == sentences


def test_tree_chunker_overlaps_consecutive_text_chunks(tmp_path: Path) -> None:
    sentences = _LONG_TEST_SENTENCES
    document = _document_with_text(tmp_path, " ".join(sentences))

    chunks = [
        node.text
        for node in TreeRagChunker(max_chunk_chars=200, overlap_sentences=1).chunk(document)
        if node.node_type is TreeNodeType.CHUNK
    ]

    assert chunks == [
        sentences[0],
        f"{sentences[0]} {sentences[1]}",
        f"{sentences[1]} {sentences[2]}",
        f"{sentences[2]} {sentences[3]}",
    ]


def test_tree_chunker_merges_caption_into_figure_chunk(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")
    section = document.sections[-1]
    figure = FigureBlock(
        page_number=4,
        text="Figure region on page 4",
        object_label="1",
    )
    caption = CaptionBlock(
        page_number=4,
        text="Figure 1: Overview of the proposed method.",
        object_label="1",
        target_type=PaperBlockType.FIGURE,
    )
    document = document.model_copy(
        update={
            "sections": [
                *document.sections[:-1],
                section.model_copy(update={"blocks": [figure, caption]}),
            ]
        }
    )

    chunks = [
        node
        for node in TreeRagChunker(max_chunk_chars=200).chunk(document)
        if node.node_type is TreeNodeType.CHUNK
    ]

    figure_chunk = next(node for node in chunks if PaperBlockType.FIGURE in node.block_types)
    assert figure_chunk.figure_caption == "Figure 1: Overview of the proposed method."
    assert "Figure 1: Overview of the proposed method." in figure_chunk.text
    assert not any(PaperBlockType.CAPTION in node.block_types for node in chunks)


def test_tree_chunker_keeps_orphan_caption_searchable(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")
    section = document.sections[-1]
    caption = CaptionBlock(
        page_number=4,
        text="Figure 9: A caption without a detected figure.",
        object_label="9",
        target_type=PaperBlockType.FIGURE,
    )
    document = document.model_copy(
        update={
            "sections": [
                *document.sections[:-1],
                section.model_copy(update={"blocks": [caption]}),
            ]
        }
    )

    chunks = [
        node
        for node in TreeRagChunker(max_chunk_chars=200).chunk(document)
        if node.node_type is TreeNodeType.CHUNK
    ]

    assert any(
        "Figure 9: A caption without a detected figure." in node.text for node in chunks
    )


def test_tree_chunker_keeps_code_block_atomic(tmp_path: Path) -> None:
    document = sample_document(tmp_path / "paper.pdf")
    section = document.sections[-1]
    code = CodeBlock(
        page_number=4,
        text="Algorithm 1: Foo\n1 x ← 1\n2 return x",
    )
    document = document.model_copy(
        update={
            "sections": [
                *document.sections[:-1],
                section.model_copy(update={"blocks": [code]}),
            ]
        }
    )

    chunks = [
        node
        for node in TreeRagChunker(max_chunk_chars=200).chunk(document)
        if node.node_type is TreeNodeType.CHUNK
    ]

    code_chunks = [node for node in chunks if PaperBlockType.CODE in node.block_types]
    assert len(code_chunks) == 1
    assert code_chunks[0].text == "Algorithm 1: Foo\n1 x ← 1\n2 return x"
    assert "\n" in code_chunks[0].text


class _FakeParser(ScientificPaperParser):
    def __init__(self, document: ParsedPaperDocument) -> None:
        self.document = document

    async def parse(
        self,
        path: str,
        *,
        paper_id: str,
        title: str | None = None,
        asset_dir: Path | None = None,
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

    async def delete_paper(self, paper_id: str) -> None:
        return None

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str] | None,
        top_k: int,
        chunks_only: bool,
        node_types: Sequence[TreeNodeType] | None = None,
        parent_ids: Sequence[str] | None = None,
    ) -> list[TreeVectorMatch]:
        return []

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        return []

    async def keyword_search(
        self,
        terms: Sequence[str],
        *,
        paper_ids: Sequence[str] | None,
        top_k: int,
        chunks_only: bool = True,
        node_types: Sequence[TreeNodeType] | None = None,
        parent_ids: Sequence[str] | None = None,
    ) -> KeywordSearchResult:
        return KeywordSearchResult()

    async def load_nodes(
        self,
        *,
        paper_ids: Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
        parent_ids: Sequence[str] | None = None,
        node_types: Sequence[TreeNodeType] | None = None,
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

    await store.delete_paper("paper")

    assert await store.load_paper_nodes(["paper"]) == []
