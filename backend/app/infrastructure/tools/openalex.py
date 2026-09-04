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

ARXIV_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)",
    re.I,
)


class OpenAlexApiError(RuntimeError):
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


class OpenAlexPaperSearchGateway(PaperSearchGateway):
    _SELECT_FIELDS = ",".join(
        (
            "id",
            "doi",
            "display_name",
            "abstract_inverted_index",
            "authorships",
            "publication_date",
            "updated_date",
            "primary_location",
            "best_oa_location",
            "ids",
        )
    )

    def __init__(
        self,
        *,
        api_url: str,
        timeout_seconds: float,
        api_key: str = "",
        min_request_interval_seconds: float = 0.1,
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
        params = {
            "search": search_input.query,
            "per_page": str(search_input.max_results),
            "select": self._SELECT_FIELDS,
            "sort": self._sort(search_input.sort_by),
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "PaperPilot/0.1 (academic research assistant)",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with self._request_lock:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                headers=headers,
                follow_redirects=True,
            ) as client:
                response = await self._request_with_retry(client, params)
        try:
            payload = cast(dict[str, Any], response.json())
            papers = [self._parse_work(item) for item in payload.get("results", [])]
        except (TypeError, ValueError, KeyError) as error:
            raise OpenAlexApiError(
                "OpenAlex returned an invalid works response",
                category="invalid_response",
                retryable=False,
            ) from error
        return PaperSearchResult(
            papers=papers,
            provider=PaperSource.OPENALEX if papers else None,
            attempts=[
                PaperSearchAttempt(
                    provider=PaperSource.OPENALEX,
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
                response = await client.get(f"{self._api_url}/works", params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise OpenAlexApiError(
                    "OpenAlex request timed out or could not connect",
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
            raise OpenAlexApiError(
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
        raise OpenAlexApiError(
            "OpenAlex request failed",
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
    def _parse_work(raw: object) -> Paper:
        item = cast(dict[str, Any], raw)
        openalex_url = str(item["id"])
        openalex_id = openalex_url.rstrip("/").rsplit("/", maxsplit=1)[-1]
        locations = [item.get("best_oa_location"), item.get("primary_location")]
        location_urls = [
            str(location.get(key))
            for location in locations
            if isinstance(location, dict)
            for key in ("landing_page_url", "pdf_url")
            if location.get(key)
        ]
        arxiv_id = next(
            (
                match.group("id")
                for url in location_urls
                if (match := ARXIV_URL_PATTERN.search(url)) is not None
            ),
            None,
        )
        primary = item.get("primary_location")
        best_oa = item.get("best_oa_location")
        landing_page = OpenAlexPaperSearchGateway._location_value(
            primary, "landing_page_url"
        ) or OpenAlexPaperSearchGateway._location_value(best_oa, "landing_page_url")
        doi = str(item.get("doi") or "")
        external_ids = {"openalex": openalex_id}
        if doi:
            external_ids["doi"] = doi.removeprefix("https://doi.org/")
        if arxiv_id:
            external_ids["arxiv"] = arxiv_id
        return Paper(
            paper_id=f"openalex:{openalex_id}",
            source=PaperSource.OPENALEX,
            arxiv_id=arxiv_id,
            external_ids=external_ids,
            title=str(item.get("display_name") or "Untitled work"),
            summary=OpenAlexPaperSearchGateway._abstract(item.get("abstract_inverted_index")),
            authors=OpenAlexPaperSearchGateway._authors(item.get("authorships")),
            published_at=OpenAlexPaperSearchGateway._date(item.get("publication_date")),
            updated_at=OpenAlexPaperSearchGateway._date(item.get("updated_date")),
            landing_page_url=HttpUrl(landing_page or openalex_url),
            pdf_url=(HttpUrl(f"https://arxiv.org/pdf/{arxiv_id}") if arxiv_id else None),
        )

    @staticmethod
    def _abstract(value: object) -> str:
        if not isinstance(value, dict):
            return ""
        positioned = [
            (int(position), str(word))
            for word, positions in value.items()
            if isinstance(positions, list)
            for position in positions
        ]
        return " ".join(word for _, word in sorted(positioned))

    @staticmethod
    def _authors(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            str(author["display_name"])
            for authorship in value
            if isinstance(authorship, dict)
            and isinstance((author := authorship.get("author")), dict)
            and author.get("display_name")
        ]

    @staticmethod
    def _location_value(value: object, key: str) -> str | None:
        return str(value[key]) if isinstance(value, dict) and value.get(key) else None

    @staticmethod
    def _date(value: object) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or UTC)

    @staticmethod
    def _sort(value: str) -> str:
        return {
            "relevance": "relevance_score:desc",
            "submittedDate": "publication_date:desc",
            "lastUpdatedDate": "updated_date:desc",
        }[value]

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
            return "OpenAlex rate limit reached (HTTP 429)"
        if status_code >= 500:
            return f"OpenAlex service is temporarily unavailable (HTTP {status_code})"
        return f"OpenAlex rejected the request (HTTP {status_code})"
