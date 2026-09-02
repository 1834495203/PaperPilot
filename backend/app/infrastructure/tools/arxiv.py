import asyncio
import re
from collections.abc import Sequence
from datetime import datetime
from time import monotonic
from xml.etree import ElementTree

import httpx
from pydantic import HttpUrl

from app.domain.papers import ArxivSearchInput, Paper
from app.domain.ports import PaperSearchGateway

ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"
ARXIV_ID_PATTERN = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.I)


class ArxivApiError(RuntimeError):
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


class ArxivPaperSearchGateway(PaperSearchGateway):
    def __init__(
        self,
        api_url: str,
        timeout_seconds: float,
        *,
        min_request_interval_seconds: float = 3.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 3.0,
    ) -> None:
        self._api_url = api_url
        self._timeout = httpx.Timeout(timeout_seconds)
        self._min_request_interval = min_request_interval_seconds
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff_seconds
        self._request_lock = asyncio.Lock()
        self._last_request_started_at: float | None = None

    async def search(self, search_input: ArxivSearchInput) -> Sequence[Paper]:
        params = self._build_params(search_input)
        headers = {"User-Agent": "PaperPilot/0.1 (academic research assistant)"}
        async with self._request_lock:
            async with httpx.AsyncClient(timeout=self._timeout, headers=headers) as client:
                response = await self._request_with_retry(client, params)

        try:
            return self._parse_feed(response.text)
        except (ElementTree.ParseError, ValueError) as error:
            raise ArxivApiError(
                "arXiv returned an invalid Atom feed",
                category="invalid_response",
                retryable=False,
            ) from error

    async def _request_with_retry(
        self,
        client: httpx.AsyncClient,
        params: dict[str, str],
    ) -> httpx.Response:
        last_error: ArxivApiError | None = None
        for attempt in range(self._max_retries + 1):
            await self._wait_for_request_slot()
            try:
                self._last_request_started_at = monotonic()
                response = await client.get(self._api_url, params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                last_error = ArxivApiError(
                    "arXiv request timed out or could not connect",
                    category="network_error",
                    retryable=True,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_backoff * (2**attempt))
                    continue
                raise last_error from error

            if response.is_success:
                return response

            retry_after = self._retry_after_seconds(response)
            retryable = response.status_code == 429 or response.status_code >= 500
            category = "rate_limited" if response.status_code == 429 else (
                "provider_error" if response.status_code >= 500 else "request_rejected"
            )
            last_error = ArxivApiError(
                self._error_message(response.status_code),
                category=category,
                retryable=retryable,
                status_code=response.status_code,
                retry_after_seconds=retry_after,
            )
            if retryable and attempt < self._max_retries:
                delay = max(retry_after or 0.0, self._retry_backoff * (2**attempt))
                await asyncio.sleep(delay)
                continue
            raise last_error

        if last_error is not None:
            raise last_error
        raise ArxivApiError(
            "arXiv request failed",
            category="unknown",
            retryable=False,
        )

    async def _wait_for_request_slot(self) -> None:
        if self._last_request_started_at is None:
            return
        remaining = self._min_request_interval - (
            monotonic() - self._last_request_started_at
        )
        if remaining > 0:
            await asyncio.sleep(remaining)

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, min(float(value), 120.0))
        except ValueError:
            return None

    @staticmethod
    def _error_message(status_code: int) -> str:
        if status_code == 429:
            return "arXiv rate limit reached (HTTP 429)"
        if status_code >= 500:
            return f"arXiv service is temporarily unavailable (HTTP {status_code})"
        return f"arXiv rejected the request (HTTP {status_code})"

    @staticmethod
    def _build_params(search_input: ArxivSearchInput) -> dict[str, str]:
        id_match = ARXIV_ID_PATTERN.search(search_input.query)
        params = {
            "start": "0",
            "max_results": str(search_input.max_results),
            "sortBy": search_input.sort_by,
            "sortOrder": "descending",
        }
        if id_match is not None:
            params["id_list"] = id_match.group(1)
        else:
            params["search_query"] = ArxivPaperSearchGateway._all_fields_query(
                search_input.query
            )
        return params

    @staticmethod
    def _all_fields_query(query: str) -> str:
        parts = re.findall(r'"([^"]+)"|(\S+)', query)
        terms = [phrase or word for phrase, word in parts]
        cleaned = [term.strip("'\";,()") for term in terms]
        scoped = [
            f'all:"{term}"' if " " in term else f"all:{term}"
            for term in cleaned
            if term and term.upper() not in {"AND", "OR", "NOT"}
        ]
        return " AND ".join(scoped) if scoped else f"all:{query.strip()}"

    @staticmethod
    def _parse_feed(xml_text: str) -> list[Paper]:
        root = ElementTree.fromstring(xml_text)
        atom = f"{{{ATOM_NAMESPACE}}}"
        papers: list[Paper] = []

        for entry in root.findall(f"{atom}entry"):
            entry_id = ArxivPaperSearchGateway._required_text(entry, f"{atom}id")
            abstract_url = entry_id.replace("http://", "https://")
            links = entry.findall(f"{atom}link")
            pdf_url = next(
                (
                    link.attrib.get("href")
                    for link in links
                    if link.attrib.get("title") == "pdf"
                ),
                None,
            )
            papers.append(
                Paper(
                    arxiv_id=entry_id.rsplit("/", maxsplit=1)[-1],
                    title=ArxivPaperSearchGateway._normalize_text(
                        ArxivPaperSearchGateway._required_text(entry, f"{atom}title")
                    ),
                    summary=ArxivPaperSearchGateway._normalize_text(
                        ArxivPaperSearchGateway._required_text(entry, f"{atom}summary")
                    ),
                    authors=[
                        ArxivPaperSearchGateway._required_text(author, f"{atom}name")
                        for author in entry.findall(f"{atom}author")
                    ],
                    published_at=datetime.fromisoformat(
                        ArxivPaperSearchGateway._required_text(entry, f"{atom}published")
                        .replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        ArxivPaperSearchGateway._required_text(entry, f"{atom}updated")
                        .replace("Z", "+00:00")
                    ),
                    abstract_url=HttpUrl(abstract_url),
                    pdf_url=HttpUrl(pdf_url) if pdf_url is not None else None,
                )
            )
        return papers

    @staticmethod
    def _required_text(parent: ElementTree.Element, path: str) -> str:
        value = parent.findtext(path)
        if value is None or not value.strip():
            raise ValueError(f"Missing required Atom field: {path}")
        return value.strip()

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.split())
