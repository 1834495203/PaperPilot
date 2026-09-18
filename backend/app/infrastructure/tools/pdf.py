import asyncio
from io import BytesIO
from urllib.parse import urlparse

import httpx
from pydantic import HttpUrl
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.domain.papers import PdfDocument
from app.domain.ports import PaperDocumentGateway


class PdfDocumentError(RuntimeError):
    pass


class ArxivPdfDocumentGateway(PaperDocumentGateway):
    """Download and extract bounded text from an arXiv PDF."""

    _MAX_REDIRECTS = 3

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_bytes: int,
        max_pages: int,
        max_characters: int,
    ) -> None:
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_bytes = max_bytes
        self._max_pages = max_pages
        self._max_characters = max_characters

    async def fetch(self, url: str) -> PdfDocument:
        source_url = self._validated_url(url)
        pdf_bytes, final_url = await self._download(source_url)
        try:
            return await asyncio.to_thread(self._extract, pdf_bytes, str(final_url))
        except (PdfReadError, ValueError, OSError) as error:
            raise PdfDocumentError(f"Unable to extract PDF text: {error}") from error

    async def _download(self, url: httpx.URL) -> tuple[bytes, httpx.URL]:
        headers = {"User-Agent": "PaperPilot/0.1 (research assistant demo)"}
        current_url = url
        async with httpx.AsyncClient(timeout=self._timeout, headers=headers) as client:
            for redirect_count in range(self._MAX_REDIRECTS + 1):
                try:
                    async with client.stream("GET", current_url) as response:
                        if response.is_redirect:
                            if redirect_count >= self._MAX_REDIRECTS:
                                raise PdfDocumentError("PDF download exceeded redirect limit")
                            location = response.headers.get("location")
                            if not location:
                                raise PdfDocumentError("PDF redirect omitted its destination")
                            current_url = self._validated_url(str(response.url.join(location)))
                            continue
                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "").lower()
                        if "pdf" not in content_type and "octet-stream" not in content_type:
                            raise PdfDocumentError(
                                f"Expected a PDF response, received {content_type or 'unknown'}"
                            )
                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > self._max_bytes:
                            raise PdfDocumentError("PDF exceeds configured download size limit")
                        chunks: list[bytes] = []
                        received = 0
                        async for chunk in response.aiter_bytes():
                            received += len(chunk)
                            if received > self._max_bytes:
                                raise PdfDocumentError("PDF exceeds configured download size limit")
                            chunks.append(chunk)
                        if not chunks:
                            raise PdfDocumentError("PDF download returned an empty body")
                        return b"".join(chunks), current_url
                except httpx.HTTPError as error:
                    raise PdfDocumentError(f"PDF download failed: {error}") from error
        raise PdfDocumentError("PDF download failed before a response was received")

    def _extract(self, pdf_bytes: bytes, source_url: str) -> PdfDocument:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PdfDocumentError("Encrypted PDF cannot be read without a password")

        page_count = len(reader.pages)
        extracted_page_limit = min(page_count, self._max_pages)
        warnings: list[str] = []
        if page_count > self._max_pages:
            warnings.append(
                f"PDF has {page_count} pages; only the first {self._max_pages} were read"
            )

        sections: list[str] = []
        remaining = self._max_characters
        extracted_pages = 0
        truncated = page_count > self._max_pages
        for page_index in range(extracted_page_limit):
            page_text = (reader.pages[page_index].extract_text() or "").strip()
            section = f"--- PAGE {page_index + 1} ---\n{page_text}\n"
            if len(section) > remaining:
                sections.append(section[:remaining])
                truncated = True
                warnings.append(
                    f"Extracted text was truncated at {self._max_characters} characters"
                )
                break
            sections.append(section)
            remaining -= len(section)
            extracted_pages += 1

        text = "\n".join(sections).strip()
        if not text or not any(character.isalnum() for character in text):
            raise PdfDocumentError(
                "PDF contains no extractable text; scanned-image OCR is not supported"
            )
        return PdfDocument(
            source_url=HttpUrl(source_url),
            page_count=page_count,
            extracted_pages=extracted_pages,
            extracted_characters=len(text),
            text=text,
            truncated=truncated,
            extraction_warnings=warnings,
        )

    @staticmethod
    def _validated_url(url: str) -> httpx.URL:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme != "https":
            raise PdfDocumentError("Only HTTPS PDF URLs are allowed")
        if hostname != "arxiv.org" and not hostname.endswith(".arxiv.org"):
            raise PdfDocumentError("Only arxiv.org PDF URLs are allowed")
        return httpx.URL(url)
