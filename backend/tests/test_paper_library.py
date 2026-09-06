from pathlib import Path
from typing import Any, cast

import pytest

from app.application.paper_library import (
    PaperLibraryService,
    PaperNotFoundError,
    PaperUploadError,
)
from app.domain.rag import (
    IndexedTreeNode,
    PaperIngestionResult,
    PaperMetadata,
    RetrievalMode,
    TreeIndexNode,
    TreeNodeType,
    TreeRetrievalReport,
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
        return PaperIngestionResult(
            paper_id=paper_id,
            metadata=PaperMetadata(
                title=title or "Parsed Paper",
                authors=["Ada Lovelace"],
                abstract="A parsed abstract.",
            ),
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
        paper_ids: list[str],
        mode: RetrievalMode,
    ) -> TreeRetrievalReport:
        return TreeRetrievalReport(
            query=query,
            mode=mode,
            paper_ids=paper_ids,
            initial_hit_count=0,
            expanded_candidate_count=0,
            hits=[],
        )


class _VectorStore:
    def __init__(self) -> None:
        self.deleted_paper_ids: list[str] = []

    async def delete_paper(self, paper_id: str) -> None:
        self.deleted_paper_ids.append(paper_id)

    async def load_paper_nodes(self, paper_ids: list[str]) -> list[IndexedTreeNode]:
        paper_id = paper_ids[0]
        assert paper_id is not None
        root = TreeIndexNode(
            node_id=f"{paper_id}:root",
            paper_id=paper_id,
            node_type=TreeNodeType.ROOT,
            title="Parsed Paper",
            children_ids=[f"{paper_id}:chunk:1"],
            level=0,
            section_path=[],
            text="Parsed Paper",
            embedding_text="Parsed Paper",
            page_start=1,
            page_end=4,
        )
        chunk = TreeIndexNode(
            node_id=f"{paper_id}:chunk:1",
            paper_id=paper_id,
            node_type=TreeNodeType.CHUNK,
            title="Method",
            parent_id=root.node_id,
            level=1,
            section_path=["1 Method"],
            text="A " + ("long method description " * 30),
            embedding_text="Method",
            page_start=2,
            page_end=2,
        )
        return [
            IndexedTreeNode(node=root, embedding=[1.0, 0.0]),
            IndexedTreeNode(node=chunk, embedding=[0.9, 0.1]),
        ]


class _FailingDeleteVectorStore(_VectorStore):
    async def delete_paper(self, paper_id: str) -> None:
        raise RuntimeError("vector database unavailable")


def _service(
    tmp_path: Path,
    *,
    vector_store: _VectorStore | None = None,
) -> PaperLibraryService:
    return PaperLibraryService(
        library_path=tmp_path / "papers",
        max_upload_bytes=1_000_000,
        ingestion=cast(Any, _Ingestion()),
        retriever=cast(Any, _Retriever()),
        vector_store=cast(Any, vector_store or _VectorStore()),
    )


@pytest.mark.asyncio
async def test_upload_persists_pdf_manifest_and_indexes_the_paper(tmp_path: Path) -> None:
    service = _service(tmp_path)
    await service.initialize()

    paper = await service.upload_pdf(
        filename="My Research.PDF",
        content=b"%PDF-1.7\nminimal-test-content",
        title="User Title",
    )

    assert paper.paper_id.startswith("my-research-")
    assert paper.title == "User Title"
    assert paper.metadata.authors == ["Ada Lovelace"]
    assert (tmp_path / "papers" / f"{paper.paper_id}.pdf").exists()
    assert [item.paper_id for item in await service.list_papers()] == [paper.paper_id]

    detail = await service.get_paper_detail(paper.paper_id)
    assert [node.node_type for node in detail.nodes] == [
        TreeNodeType.ROOT,
        TreeNodeType.CHUNK,
    ]
    assert detail.nodes[1].text_preview.endswith("…")


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf_content(tmp_path: Path) -> None:
    service = _service(tmp_path)
    await service.initialize()

    with pytest.raises(PaperUploadError, match="PDF signature"):
        await service.upload_pdf(filename="fake.pdf", content=b"not a pdf")


@pytest.mark.asyncio
async def test_delete_removes_pdf_manifest_and_vector_nodes(tmp_path: Path) -> None:
    vector_store = _VectorStore()
    service = _service(tmp_path, vector_store=vector_store)
    await service.initialize()
    paper = await service.upload_pdf(
        filename="delete-me.pdf",
        content=b"%PDF-1.7\nminimal-test-content",
    )

    deleted = await service.delete_paper(paper.paper_id)

    assert deleted.paper_id == paper.paper_id
    assert vector_store.deleted_paper_ids == [paper.paper_id]
    assert not (tmp_path / "papers" / f"{paper.paper_id}.pdf").exists()
    assert not (tmp_path / "papers" / f"{paper.paper_id}.json").exists()
    with pytest.raises(PaperNotFoundError):
        await service.get_paper(paper.paper_id)


@pytest.mark.asyncio
async def test_delete_restores_files_when_vector_deletion_fails(tmp_path: Path) -> None:
    service = _service(tmp_path, vector_store=_FailingDeleteVectorStore())
    await service.initialize()
    paper = await service.upload_pdf(
        filename="keep-me.pdf",
        content=b"%PDF-1.7\nminimal-test-content",
    )

    with pytest.raises(RuntimeError, match="vector database unavailable"):
        await service.delete_paper(paper.paper_id)

    assert (tmp_path / "papers" / f"{paper.paper_id}.pdf").exists()
    assert (tmp_path / "papers" / f"{paper.paper_id}.json").exists()
    assert (await service.get_paper(paper.paper_id)).paper_id == paper.paper_id


@pytest.mark.asyncio
async def test_retrieval_rejects_unknown_library_paper(tmp_path: Path) -> None:
    service = _service(tmp_path)
    await service.initialize()

    with pytest.raises(PaperNotFoundError, match="missing"):
        await service.retrieve(
            "query",
            paper_ids=["missing"],
            mode=RetrievalMode.FACT,
        )


def test_get_asset_path_validates_and_resolves(tmp_path: Path) -> None:
    service = _service(tmp_path)

    good = service.get_asset_path("paper-abc123", "fig-0001-00.png")
    assert good == (tmp_path / "papers" / "paper-abc123.assets" / "fig-0001-00.png").resolve()

    with pytest.raises(PaperNotFoundError):
        service.get_asset_path("paper-abc123", "../secret.png")
    with pytest.raises(PaperNotFoundError):
        service.get_asset_path("paper-abc123", "sub/dir.png")
    with pytest.raises(PaperNotFoundError):
        service.get_asset_path("bad paper id!", "fig.png")


def test_get_pdf_path_resolves_managed_pdf(tmp_path: Path) -> None:
    service = _service(tmp_path)
    pdf = tmp_path / "papers" / "paper-abc123.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7\n")

    resolved = service.get_pdf_path("paper-abc123")
    assert resolved.name == "paper-abc123.pdf"
    assert resolved.is_file()

    with pytest.raises(PaperNotFoundError):
        service.get_pdf_path("missing-paper")
    with pytest.raises(PaperNotFoundError):
        service.get_pdf_path("bad id!")
