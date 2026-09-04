import asyncio
import re
from datetime import UTC, datetime
from time import monotonic
from typing import Any, cast

import httpx
from pydantic import HttpUrl

from app.domain.papers import (
    Paper,
    PaperSearchAttempt,
    PaperSearchInput,
    PaperSearchResult,
    PaperSource,
)
from app.domain.ports import PaperSearchGateway


class SemanticScholarApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class SemanticScholarPaperSearchGateway(PaperSearchGateway):
    _FIELDS = ",".join(
        (
            "paperId",
            "externalIds",
            "url",
            "title",
            "abstract",
            "authors",
            "publicationDate",
            "year",
            "openAccessPdf",
        )
    )

    def __init__(
        self,
        *,
        api_url: str,
        timeout_seconds: float,
        api_key: str = "",
        min_request_interval_seconds: float = 1.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._api_key = api_key
        self._min_request_interval = min_request_interval_seconds
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_seconds
        self._request_lock = asyncio.Lock()
        self._last_request_started_at: float | None = None

    async def search(self, search_input: PaperSearchInput) -> PaperSearchResult:
        query = re.sub(r"(?<=\w)-(?=\w)", " ", search_input.query)
        params = {
            "query": query,
            "limit": str(search_input.max_results),
            "fields": self._FIELDS,
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "PaperPilot/0.1 (academic research assistant)",
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        async with self._request_lock:
            async with httpx.AsyncClient(timeout=self._timeout, headers=headers) as client:
                response = await self._request_with_retry(client, params)
        try:
            payload = cast(dict[str, Any], response.json())
            papers = [self._parse_paper(item) for item in payload.get("data", [])]
        except (TypeError, ValueError, KeyError) as error:
            raise SemanticScholarApiError(
                "Semantic Scholar returned an invalid paper response",
                category="invalid_response",
                retryable=False,
            ) from error
        return PaperSearchResult(
            papers=papers,
            provider=PaperSource.SEMANTIC_SCHOLAR if papers else None,
            attempts=[
                PaperSearchAttempt(
                    provider=PaperSource.SEMANTIC_SCHOLAR,
                    status="completed" if papers else "empty",
                    result_count=len(papers),
                )
            ],
        )

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str],
    ) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            await self._wait_for_request_slot()
            try:
                self._last_request_started_at = monotonic()
                response = await client.get(
                    f"{self._api_url}/paper/search",
                    params=params,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise SemanticScholarApiError(
                    "Semantic Scholar request timed out or could not connect",
                    category="network_error",
                    retryable=True,
                ) from error
            if response.is_success:
                return response
            retry_after = self._retry_after_seconds(response)
            retryable = response.status_code == 429 or response.status_code >= 500
            if retryable and attempt < self._max_retries:
                await asyncio.sleep(max(retry_after or 0.0, self._retry_backoff * (2**attempt)))
                continue
            raise SemanticScholarApiError(
                self._error_message(response.status_code),
                category=(
                    "rate_limited"
                    if response.status_code == 429
                    else "provider_error"
                    if response.status_code >= 500
                    else "request_rejected"
                ),
                retryable=retryable,
                status_code=response.status_code,
                retry_after_seconds=retry_after,
            )
        raise SemanticScholarApiError(
            "Semantic Scholar request failed",
            category="unknown",
            retryable=False,
        )

    async def _wait_for_request_slot(self) -> None:
        if self._last_request_started_at is None:
            return
        remaining = self._min_request_interval - (monotonic() - self._last_request_started_at)
        if remaining > 0:
            await asyncio.sleep(remaining)

    @staticmethod
    def _parse_paper(raw: object) -> Paper:
        item = cast(dict[str, Any], raw)
        paper_id = str(item["paperId"])
        raw_ids = item.get("externalIds")
        external_ids = (
            {str(key).lower(): str(value) for key, value in raw_ids.items() if value}
            if isinstance(raw_ids, dict)
            else {}
        )
        arxiv_id = external_ids.get("arxiv")
        published_at = SemanticScholarPaperSearchGateway._publication_date(item)
        return Paper(
            paper_id=f"semantic_scholar:{paper_id}",
            source=PaperSource.SEMANTIC_SCHOLAR,
            arxiv_id=arxiv_id,
            external_ids={"semantic_scholar": paper_id, **external_ids},
            title=str(item.get("title") or "Untitled paper"),
            summary=str(item.get("abstract") or ""),
            authors=[
                str(author["name"])
                for author in item.get("authors", [])
                if isinstance(author, dict) and author.get("name")
            ],
            published_at=published_at,
            updated_at=published_at,
            landing_page_url=HttpUrl(
                str(item.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}")
            ),
            pdf_url=(HttpUrl(f"https://arxiv.org/pdf/{arxiv_id}") if arxiv_id else None),
        )

    @staticmethod
    def _publication_date(item: dict[str, Any]) -> datetime | None:
        value = item.get("publicationDate")
        if value:
            parsed = datetime.fromisoformat(str(value))
            return parsed.replace(tzinfo=UTC)
        year = item.get("year")
        return datetime(int(year), 1, 1, tzinfo=UTC) if year else None

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        try:
            return max(0.0, min(float(value), 120.0)) if value is not None else None
        except ValueError:
            return None

    @staticmethod
    def _error_message(status_code: int) -> str:
        if status_code == 429:
            return "Semantic Scholar rate limit reached (HTTP 429)"
        if status_code >= 500:
            return f"Semantic Scholar is temporarily unavailable (HTTP {status_code})"
        return f"Semantic Scholar rejected the request (HTTP {status_code})"
