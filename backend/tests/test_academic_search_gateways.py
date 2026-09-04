from typing import cast
from unittest.mock import AsyncMock, create_autospec

import pytest

from app.domain.papers import (
    Paper,
    PaperSearchAttempt,
    PaperSearchInput,
    PaperSearchResult,
    PaperSource,
)
from app.domain.ports import PaperSearchGateway
from app.infrastructure.tools.openalex import OpenAlexPaperSearchGateway
from app.infrastructure.tools.paper_search import (
    FallbackPaperSearchGateway,
    SearchProvider,
)
from app.infrastructure.tools.semantic_scholar import (
    SemanticScholarPaperSearchGateway,
)


def make_result(source: PaperSource, paper_id: str | None = None) -> PaperSearchResult:
    papers = (
        [
            Paper(
                paper_id=f"{source.value}:{paper_id}",
                source=source,
                external_ids={source.value: paper_id},
                title="A relevant paper",
                summary="Abstract",
                authors=["Ada Example"],
                landing_page_url="https://example.com/paper",
                pdf_url=None,
            )
        ]
        if paper_id is not None
        else []
    )
    return PaperSearchResult(
        papers=papers,
        provider=source if papers else None,
        attempts=[
            PaperSearchAttempt(
                provider=source,
                status="completed" if papers else "empty",
                result_count=len(papers),
            )
        ],
    )


def gateway_mock() -> tuple[PaperSearchGateway, AsyncMock]:
    gateway = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    return gateway, cast(AsyncMock, gateway.search)


@pytest.mark.asyncio
async def test_fallback_stops_after_openalex_returns_results() -> None:
    openalex, openalex_search = gateway_mock()
    semantic, semantic_search = gateway_mock()
    arxiv, arxiv_search = gateway_mock()
    openalex_search.return_value = make_result(PaperSource.OPENALEX, "W1")
    gateway = FallbackPaperSearchGateway(
        [
            SearchProvider(PaperSource.OPENALEX, openalex),
            SearchProvider(PaperSource.SEMANTIC_SCHOLAR, semantic),
            SearchProvider(PaperSource.ARXIV, arxiv),
        ]
    )

    result = await gateway.search(PaperSearchInput(query="agentic RAG"))

    assert result.provider is PaperSource.OPENALEX
    openalex_search.assert_awaited_once()
    semantic_search.assert_not_awaited()
    arxiv_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_fallback_uses_semantic_scholar_then_arxiv_on_empty_results() -> None:
    openalex, openalex_search = gateway_mock()
    semantic, semantic_search = gateway_mock()
    arxiv, arxiv_search = gateway_mock()
    openalex_search.return_value = make_result(PaperSource.OPENALEX)
    semantic_search.return_value = make_result(PaperSource.SEMANTIC_SCHOLAR)
    arxiv_search.return_value = make_result(PaperSource.ARXIV, "2401.00001")
    gateway = FallbackPaperSearchGateway(
        [
            SearchProvider(PaperSource.OPENALEX, openalex),
            SearchProvider(PaperSource.SEMANTIC_SCHOLAR, semantic),
            SearchProvider(PaperSource.ARXIV, arxiv),
        ]
    )

    result = await gateway.search(PaperSearchInput(query="agentic RAG"))

    assert result.provider is PaperSource.ARXIV
    assert [attempt.provider for attempt in result.attempts] == [
        PaperSource.OPENALEX,
        PaperSource.SEMANTIC_SCHOLAR,
        PaperSource.ARXIV,
    ]
    openalex_search.assert_awaited_once()
    semantic_search.assert_awaited_once()
    arxiv_search.assert_awaited_once()


@pytest.mark.asyncio
async def test_fallback_continues_when_openalex_fails() -> None:
    openalex, openalex_search = gateway_mock()
    semantic, semantic_search = gateway_mock()
    openalex_search.side_effect = RuntimeError("OpenAlex unavailable")
    semantic_search.return_value = make_result(PaperSource.SEMANTIC_SCHOLAR, "S1")
    gateway = FallbackPaperSearchGateway(
        [
            SearchProvider(PaperSource.OPENALEX, openalex),
            SearchProvider(PaperSource.SEMANTIC_SCHOLAR, semantic),
        ]
    )

    result = await gateway.search(PaperSearchInput(query="agentic RAG"))

    assert result.provider is PaperSource.SEMANTIC_SCHOLAR
    assert result.attempts[0].status == "failed"
    assert result.attempts[1].status == "completed"


def test_openalex_parser_reconstructs_abstract_and_arxiv_link() -> None:
    paper = OpenAlexPaperSearchGateway._parse_work(
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1000/example",
            "display_name": "Tree Retrieval",
            "abstract_inverted_index": {"Tree": [0], "retrieval": [1]},
            "authorships": [{"author": {"display_name": "Ada Example"}}],
            "publication_date": "2024-01-02",
            "updated_date": "2024-02-03T00:00:00Z",
            "primary_location": {"landing_page_url": "https://arxiv.org/abs/2401.12345v2"},
            "best_oa_location": None,
        }
    )

    assert paper.paper_id == "openalex:W123"
    assert paper.summary == "Tree retrieval"
    assert paper.arxiv_id == "2401.12345v2"
    assert str(paper.pdf_url) == "https://arxiv.org/pdf/2401.12345v2"


def test_semantic_scholar_parser_keeps_generic_identity() -> None:
    paper = SemanticScholarPaperSearchGateway._parse_paper(
        {
            "paperId": "abc123",
            "externalIds": {"DOI": "10.1000/example"},
            "url": "https://www.semanticscholar.org/paper/abc123",
            "title": "Semantic Search",
            "abstract": "An abstract",
            "authors": [{"name": "Grace Example"}],
            "publicationDate": "2025-04-05",
            "openAccessPdf": {"url": "https://example.org/paper.pdf"},
        }
    )

    assert paper.paper_id == "semantic_scholar:abc123"
    assert paper.external_ids["doi"] == "10.1000/example"
    assert paper.arxiv_id is None
    assert paper.pdf_url is None
