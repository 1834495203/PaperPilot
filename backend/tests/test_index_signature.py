from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from app.application.paper_library import PaperLibraryService
from app.domain.rag import (
    IndexedTreeNode,
    PageTextBlock,
    PaperIndexStatus,
    PaperIngestionResult,
    PaperMetadata,
    PaperSection,
    ParsedPaperDocument,
    RetrievalMode,
    TreeIndexNode,
    TreeNodeType,
    TreeRetrievalReport,
)
from app.infrastructure.rag.chroma_store import (
    ChromaTreeVectorStore,
    IndexSignatureMismatchError,
)
from app.infrastructure.rag.index_signature import build_index_signature
from app.infrastructure.rag.tree_chunker import TreeRagChunker


def _document(paper_id: str = "paper") -> ParsedPaperDocument:
    return ParsedPaperDocument(
        paper_id=paper_id,
        metadata=PaperMetadata(title="Keyword Channel Paper"),
        source_path=Path("paper.pdf"),
        page_count=1,
        sections=[
            PaperSection(
                section_id=f"{paper_id}:section:0001",
                index="1",
                title="Introduction",
                semantic_role="introduction",
                level=1,
                blocks=[
                    PageTextBlock(
                        block_id=f"{paper_id}:block:1",
                        page_number=1,
                        text="The system retrieves passages and returns citations.",
                    )
                ],
                page_start=1,
                page_end=1,
            )
        ],
    )


class _Ingestion:
    async def ingest_pdf(
        self,
        path: str,
        *,
        paper_id: str,
        title: str | None = None,
        asset_dir: Path | None = None,
    ) -> PaperIngestionResult:
        del path, asset_dir
        return PaperIngestionResult(
            paper_id=paper_id,
            metadata=PaperMetadata(title=title or "Parsed Paper", authors=["Ada Lovelace"]),
            page_count=4,
            section_count=3,
            node_count=8,
            chunk_count=4,
            vector_collection="test_tree",
        )


class _Retriever:
    async def retrieve(
        self,
        query: str,
        *,
        paper_ids: list[str] | None = None,
        mode: RetrievalMode,
        **kwargs: object,
    ) -> TreeRetrievalReport:
        del kwargs
        return TreeRetrievalReport(
            query=query,
            mode=mode,
            paper_ids=paper_ids or [],
            initial_hit_count=0,
            expanded_candidate_count=0,
            hits=[],
        )


class _VectorStore:
    async def delete_paper(self, paper_id: str) -> None:
        return None

    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]:
        paper_id = paper_ids[0]
        node = TreeIndexNode(
            node_id=f"{paper_id}:root",
            paper_id=paper_id,
            node_type=TreeNodeType.ROOT,
            title="Parsed Paper",
            children_ids=[],
            level=0,
            section_path=[],
            text="Parsed Paper",
            embedding_text="Parsed Paper",
        )
        return [IndexedTreeNode(node=node, embedding=[1.0, 0.0])]


def _library(tmp_path: Path, *, embedding_model: str) -> PaperLibraryService:
    return PaperLibraryService(
        library_path=tmp_path / "papers",
        max_upload_bytes=1_000_000,
        ingestion=cast(Any, _Ingestion()),
        retriever=cast(Any, _Retriever()),
        vector_store=cast(Any, _VectorStore()),
        index_signature=build_index_signature(
            embedding_model=embedding_model,
            embedding_dimensions=2,
        ),
    )


def test_signature_fingerprint_tracks_model_and_code_versions() -> None:
    base = build_index_signature(embedding_model="model-a", embedding_dimensions=768)
    same = build_index_signature(embedding_model="model-a", embedding_dimensions=768)
    other_model = build_index_signature(
        embedding_model="model-b", embedding_dimensions=768
    )
    other_dimensions = build_index_signature(
        embedding_model="model-a", embedding_dimensions=1024
    )

    assert base.fingerprint == same.fingerprint
    assert base.fingerprint != other_model.fingerprint
    assert base.fingerprint != other_dimensions.fingerprint
    assert base.chunker_version == TreeRagChunker.VERSION
    assert base.schema_version == ChromaTreeVectorStore.SCHEMA_VERSION


def test_index_status_only_flags_a_recorded_mismatch() -> None:
    current = build_index_signature(embedding_model="model-a", embedding_dimensions=None)
    matching = PaperIndexStatus(
        paper_id="paper-a",
        title="Paper A",
        chunk_count=4,
        index_signature=current.fingerprint,
        current_signature=current.fingerprint,
    )
    stale = matching.model_copy(update={"index_signature": "0000deadbeef0000"})
    unknown = matching.model_copy(update={"index_signature": None})

    assert matching.needs_rebuild is False
    assert matching.fingerprint_recorded is True
    assert stale.needs_rebuild is True
    assert unknown.needs_rebuild is False
    assert unknown.fingerprint_recorded is False


@pytest.mark.asyncio
async def test_chroma_store_records_the_index_signature(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    signature = build_index_signature(embedding_model="model-a", embedding_dimensions=3)
    store = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="signature_test",
        index_signature=signature,
    )
    document = _document()
    nodes = TreeRagChunker(max_chunk_chars=200).chunk(document)

    await store.replace_paper(
        document,
        nodes,
        [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(nodes)],
    )

    reopened = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="signature_test",
        index_signature=signature,
    )
    changed = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="signature_test",
        index_signature=build_index_signature(
            embedding_model="model-b", embedding_dimensions=3
        ),
    )

    assert store.stored_index_signature == signature.fingerprint
    assert reopened.index_signature_matches is True
    assert changed.index_signature_matches is False
    assert changed.stored_index_signature == signature.fingerprint
    assert await reopened.load_nodes(node_ids=[nodes[0].node_id])


@pytest.mark.asyncio
async def test_chroma_keyword_channel_finds_text_beyond_vector_recall(
    tmp_path: Path,
) -> None:
    pytest.importorskip("chromadb")
    store = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="keyword_test",
    )
    document = _document()
    nodes = TreeRagChunker(max_chunk_chars=200).chunk(document)

    await store.replace_paper(
        document,
        nodes,
        [[1.0, 0.0, 0.5] for _ in nodes],
    )

    result = await store.keyword_search(
        ["retrieves"],
        paper_ids=["paper"],
        top_k=5,
    )
    empty = await store.keyword_search(["unfindableterm"], paper_ids=["paper"], top_k=5)

    assert result.matches
    assert result.pool_size == 1
    assert result.truncated_terms == []
    assert result.matches[0].keyword_score == 1.0
    assert result.matches[0].node.node_type is TreeNodeType.CHUNK
    assert "retrieves" in result.matches[0].node.text
    assert result.matches[0].matched_terms == ["retrieves"]
    assert empty.matches == []
    assert empty.pool_size == 0


@pytest.mark.asyncio
async def test_library_records_and_reports_the_index_signature(tmp_path: Path) -> None:
    service = _library(tmp_path, embedding_model="model-a")
    await service.initialize()

    assert await service.index_status() == []
    assert service.index_signature is not None

    paper = await service.upload_pdf(filename="paper.pdf", content=b"%PDF-1.4 body")
    statuses = await service.index_status()

    assert paper.index_signature == service.index_signature.fingerprint
    assert len(statuses) == 1
    assert statuses[0].needs_rebuild is False
    assert statuses[0].current_signature == service.index_signature.fingerprint

    stale = await _library(tmp_path, embedding_model="model-b").index_status()

    assert stale[0].needs_rebuild is True
    assert stale[0].index_signature == service.index_signature.fingerprint


@pytest.mark.asyncio
async def test_bm25_uses_real_term_frequency(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    store = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="bm25_test",
    )
    document = ParsedPaperDocument(
        paper_id="paper",
        metadata=PaperMetadata(title="Term Frequency Paper"),
        source_path=Path("paper.pdf"),
        page_count=1,
        sections=[
            PaperSection(
                section_id=f"paper:section:{index:04d}",
                index=str(index + 1),
                title=heading,
                semantic_role="method",
                level=1,
                blocks=[
                    PageTextBlock(
                        block_id=f"paper:block:{index}",
                        page_number=1,
                        text=text,
                    )
                ],
                page_start=1,
                page_end=1,
            )
            for index, (heading, text) in enumerate(
                [
                    ("1 Once", "This section mentions latency a single time."),
                    (
                        "2 Repeated",
                        "Latency latency latency latency latency is measured repeatedly.",
                    ),
                ]
            )
        ],
    )
    nodes = TreeRagChunker(max_chunk_chars=400).chunk(document)
    await store.replace_paper(document, nodes, [[1.0, 0.0] for _ in nodes])

    result = await store.keyword_search(["latency"], paper_ids=["paper"], top_k=5)

    assert len(result.matches) == 2
    repeated, single = result.matches
    assert "repeatedly" in repeated.node.text
    assert repeated.keyword_score == 1.0
    assert single.keyword_score < repeated.keyword_score


@pytest.mark.asyncio
async def test_keyword_pool_truncation_is_reported(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    store = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="truncation_test",
        keyword_filter_limit=1,
    )
    document = _document()
    second = document.sections[0].model_copy(
        update={"section_id": "paper:section:0002", "index": "2", "title": "Second"}
    )
    document = document.model_copy(update={"sections": [*document.sections, second]})
    nodes = TreeRagChunker(max_chunk_chars=200).chunk(document)
    await store.replace_paper(document, nodes, [[1.0, 0.0, 0.5] for _ in nodes])

    result = await store.keyword_search(["retrieves"], paper_ids=["paper"], top_k=5)

    assert result.truncated_terms == ["retrieves"]
    assert result.pool_size == 1
    assert result.matches[0].keyword_score == 1.0


@pytest.mark.asyncio
async def test_mismatched_index_signature_is_enforced(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    signature = build_index_signature(embedding_model="model-a", embedding_dimensions=3)
    store = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="enforcement_test",
        index_signature=signature,
    )
    document = _document()
    nodes = TreeRagChunker(max_chunk_chars=200).chunk(document)
    await store.replace_paper(document, nodes, [[1.0, 0.0, 0.5] for _ in nodes])

    other = build_index_signature(embedding_model="model-b", embedding_dimensions=3)
    strict = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="enforcement_test",
        index_signature=other,
    )
    relaxed = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="enforcement_test",
        index_signature=other,
        enforce_index_signature=False,
    )

    with pytest.raises(IndexSignatureMismatchError):
        await strict.replace_paper(document, nodes, [[1.0, 0.0, 0.5] for _ in nodes])
    with pytest.raises(IndexSignatureMismatchError):
        await strict.similarity_search(
            [1.0, 0.0, 0.5],
            paper_ids=["paper"],
            top_k=1,
            chunks_only=True,
        )
    with pytest.raises(IndexSignatureMismatchError):
        await strict.load_nodes(node_ids=[nodes[0].node_id])

    matches = await relaxed.similarity_search(
        [1.0, 0.0, 0.5],
        paper_ids=["paper"],
        top_k=1,
        chunks_only=True,
    )
    assert matches
    assert relaxed.index_signature_matches is False


@pytest.mark.asyncio
async def test_legacy_collection_without_a_fingerprint_keeps_working(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    legacy = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="legacy_test",
    )
    document = _document()
    nodes = TreeRagChunker(max_chunk_chars=200).chunk(document)
    await legacy.replace_paper(document, nodes, [[1.0, 0.0, 0.5] for _ in nodes])

    configured = ChromaTreeVectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="legacy_test",
        index_signature=build_index_signature(
            embedding_model="model-a", embedding_dimensions=3
        ),
    )

    assert configured.stored_index_signature is None
    assert configured.index_signature_matches is True
    assert await configured.load_nodes(node_ids=[nodes[0].node_id])
