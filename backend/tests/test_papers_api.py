from datetime import UTC, datetime
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_paper_library
from app.api.routes import papers
from app.application.paper_library import PaperLibraryService
from app.domain.rag import IndexedPaper, PaperMetadata


class _Library:
    max_upload_bytes = 1_000_000

    def __init__(self) -> None:
        self.paper = IndexedPaper(
            paper_id="uploaded-abc123",
            metadata=PaperMetadata(
                title="Uploaded Paper",
                authors=["Ada Lovelace"],
                abstract="An abstract.",
            ),
            original_filename="paper.pdf",
            content_sha256="a" * 64,
            page_count=5,
            section_count=4,
            node_count=10,
            chunk_count=5,
            created_at=datetime.now(UTC),
        )

    async def upload_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        title: str | None = None,
    ) -> IndexedPaper:
        assert filename == "paper.pdf"
        assert content.startswith(b"%PDF-")
        return self.paper

    async def list_papers(self) -> list[IndexedPaper]:
        return [self.paper]


def _client() -> TestClient:
    application = FastAPI()
    application.include_router(papers.router, prefix="/api/v1")
    library = _Library()
    application.dependency_overrides[get_paper_library] = lambda: cast(
        PaperLibraryService,
        cast(Any, library),
    )
    return TestClient(application)


def test_upload_and_list_papers_api() -> None:
    with _client() as client:
        uploaded = client.post(
            "/api/v1/papers",
            files={"file": ("paper.pdf", b"%PDF-1.7 fake", "application/pdf")},
        )
        listed = client.get("/api/v1/papers")

    assert uploaded.status_code == 201
    assert uploaded.json()["paper_id"] == "uploaded-abc123"
    assert listed.status_code == 200
    assert listed.json()[0]["title"] == "Uploaded Paper"
    assert listed.json()[0]["authors"] == ["Ada Lovelace"]
