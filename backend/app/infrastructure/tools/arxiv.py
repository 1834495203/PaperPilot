from collections.abc import Sequence
from datetime import datetime
from xml.etree import ElementTree

import httpx
from pydantic import HttpUrl

from app.domain.papers import ArxivSearchInput, Paper
from app.domain.ports import PaperSearchGateway

ATOM_NAMESPACE = "http://www.w3.org/2005/Atom"
ARXIV_NAMESPACE = "http://arxiv.org/schemas/atom"


class ArxivApiError(RuntimeError):
    pass


class ArxivPaperSearchGateway(PaperSearchGateway):
    def __init__(self, api_url: str, timeout_seconds: float) -> None:
        self._api_url = api_url
        self._timeout = httpx.Timeout(timeout_seconds)

    async def search(self, search_input: ArxivSearchInput) -> Sequence[Paper]:
        params = {
            "search_query": f"all:{search_input.query}",
            "start": "0",
            "max_results": str(search_input.max_results),
            "sortBy": search_input.sort_by,
            "sortOrder": "descending",
        }
        headers = {"User-Agent": "PaperPilot/0.1 (research assistant demo)"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout, headers=headers) as client:
                response = await client.get(self._api_url, params=params)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise ArxivApiError(f"arXiv request failed: {error}") from error

        try:
            return self._parse_feed(response.text)
        except (ElementTree.ParseError, ValueError) as error:
            raise ArxivApiError("arXiv returned an invalid Atom feed") from error

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
