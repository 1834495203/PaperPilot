import httpx
import pytest

from app.domain.papers import PaperSearchInput
from app.infrastructure.tools.arxiv import ArxivApiError, ArxivPaperSearchGateway


def test_parse_feed_returns_typed_paper() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.12345v1</id>
        <updated>2024-01-20T10:00:00Z</updated>
        <published>2024-01-18T10:00:00Z</published>
        <title>  A Useful   Research Paper </title>
        <summary> First line.\n Second line. </summary>
        <author><name>Ada Lovelace</name></author>
        <link title="pdf" href="https://arxiv.org/pdf/2401.12345v1" />
      </entry>
    </feed>"""

    papers = ArxivPaperSearchGateway._parse_feed(feed)

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.12345v1"
    assert papers[0].title == "A Useful Research Paper"
    assert papers[0].summary == "First line. Second line."
    assert papers[0].authors == ["Ada Lovelace"]


def test_known_arxiv_id_uses_exact_id_list() -> None:
    params = ArxivPaperSearchGateway._build_params(
        PaperSearchInput(query="Read arXiv:2409.13740 and summarize it", max_results=10)
    )

    assert params["id_list"] == "2409.13740"
    assert "search_query" not in params


def test_keyword_query_scopes_every_term() -> None:
    params = ArxivPaperSearchGateway._build_params(
        PaperSearchInput(query='PaperQA "scientific literature QA"', max_results=5)
    )

    assert params["search_query"] == 'all:PaperQA AND all:"scientific literature QA"'


@pytest.mark.asyncio
async def test_rate_limit_is_retried_once_and_then_succeeds() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, text="<feed />"),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        response = next(responses)
        response.request = request
        return response

    gateway = ArxivPaperSearchGateway(
        "https://export.arxiv.org/api/query",
        5,
        min_request_interval_seconds=0,
        max_retries=1,
        retry_backoff_seconds=0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await gateway._request_with_retry(client, {"search_query": "all:test"})

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_final_rate_limit_exposes_structured_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "12"},
            request=request,
        )

    gateway = ArxivPaperSearchGateway(
        "https://export.arxiv.org/api/query",
        5,
        min_request_interval_seconds=0,
        max_retries=0,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ArxivApiError) as raised:
            await gateway._request_with_retry(client, {"search_query": "all:test"})

    assert raised.value.category == "rate_limited"
    assert raised.value.retryable is True
    assert raised.value.status_code == 429
    assert raised.value.retry_after_seconds == 12
