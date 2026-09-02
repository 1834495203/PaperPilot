import asyncio
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PageObject, PdfReader
from pypdf.errors import PdfReadError

from app.domain.ports import ScientificPaperParser
from app.domain.rag import PageTextBlock, PaperMetadata, PaperSection, ParsedPaperDocument


class ScientificPdfParseError(RuntimeError):
    pass


@dataclass(slots=True)
class _SectionBuilder:
    section_id: str
    index: str | None
    title: str
    level: int
    parent_section_id: str | None
    heading_page: int
    blocks: list[PageTextBlock] = field(default_factory=list)

    def build(self) -> PaperSection:
        page_end = max(
            (block.page_number for block in self.blocks),
            default=self.heading_page,
        )
        return PaperSection(
            section_id=self.section_id,
            index=self.index,
            title=self.title,
            level=self.level,
            parent_section_id=self.parent_section_id,
            blocks=self.blocks,
            page_start=self.heading_page,
            page_end=page_end,
        )


class PypdfScientificPaperParser(ScientificPaperParser):
    """Extract page-aware scientific-paper sections while preserving native headings."""

    _NUMBERED_HEADING = re.compile(
        r"^(?P<index>(?:\d+|[A-H])(?:\.\d+){0,4})[.)]?\s+(?P<title>\S.{1,119})$"
    )
    _NAMED_HEADINGS = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "method",
        "methods",
        "methodology",
        "experiments",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "acknowledgments",
        "references",
        "appendix",
    }

    async def parse(
        self,
        path: str,
        *,
        paper_id: str,
        title: str | None = None,
    ) -> ParsedPaperDocument:
        return await asyncio.to_thread(
            self._parse_sync,
            Path(path),
            paper_id,
            title,
        )

    def _parse_sync(
        self,
        path: Path,
        paper_id: str,
        title: str | None,
    ) -> ParsedPaperDocument:
        if path.suffix.lower() != ".pdf":
            raise ScientificPdfParseError("Only PDF documents can be ingested")
        if not path.is_file():
            raise ScientificPdfParseError(f"PDF does not exist: {path}")
        try:
            reader = PdfReader(path, strict=False)
            if reader.is_encrypted and reader.decrypt("") == 0:
                raise ScientificPdfParseError("Encrypted PDF requires a password")
            page_texts = [self._extract_page_text(page) for page in reader.pages]
            first_page_plain_text = str(reader.pages[0].extract_text() or "")
            pdf_metadata = dict(reader.metadata or {})
        except (PdfReadError, OSError, ValueError) as error:
            raise ScientificPdfParseError(f"Unable to parse PDF: {error}") from error
        if not any(text.strip() for text in page_texts):
            raise ScientificPdfParseError("PDF contains no extractable text; OCR is not supported")

        repeated_lines = self._repeated_page_lines(page_texts)
        sections = self._extract_sections(paper_id, page_texts, repeated_lines)
        metadata = self._extract_metadata(
            page_texts,
            sections,
            pdf_metadata,
            supplied_title=title,
            fallback_title=path.stem,
            identifier_text=first_page_plain_text,
        )
        return ParsedPaperDocument(
            paper_id=paper_id,
            metadata=metadata,
            source_path=path.resolve(),
            page_count=len(page_texts),
            sections=sections,
        )

    @staticmethod
    def _extract_page_text(page: PageObject) -> str:
        extract_text = page.extract_text
        try:
            layout_text = str(extract_text(extraction_mode="layout") or "")
        except TypeError:
            layout_text = ""
        plain_text = str(extract_text() or "")
        return max((layout_text, plain_text), key=len)

    @staticmethod
    def _repeated_page_lines(page_texts: list[str]) -> set[str]:
        counts: Counter[str] = Counter()
        for text in page_texts:
            unique_lines = {
                line.strip()
                for line in text.splitlines()
                if 3 <= len(line.strip()) <= 120
            }
            counts.update(unique_lines)
        threshold = max(2, math.ceil(len(page_texts) * 0.6))
        return {line for line, count in counts.items() if count >= threshold}

    def _extract_sections(
        self,
        paper_id: str,
        page_texts: list[str],
        repeated_lines: set[str],
    ) -> list[PaperSection]:
        builders: list[_SectionBuilder] = []
        current: _SectionBuilder | None = None
        buffer: list[str] = []
        level_stack: dict[int, str] = {}
        has_numbered_headings = any(
            self._valid_numbered_heading(line.strip()) is not None
            for text in page_texts
            for line in text.splitlines()
        )

        def flush(page_number: int) -> None:
            nonlocal buffer
            if current is not None:
                text = self._normalize_body_lines(buffer)
                if text:
                    current.blocks.append(PageTextBlock(page_number=page_number, text=text))
            buffer = []

        for page_number, page_text in enumerate(page_texts, start=1):
            for raw_line in page_text.splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()
                if line in repeated_lines:
                    continue
                heading = self._parse_heading(line, allow_named=not has_numbered_headings)
                if heading is None:
                    buffer.append(line)
                    continue
                flush(page_number)
                index, heading_title, level = heading
                parent_id = level_stack.get(level - 1)
                section_id = f"{paper_id}:section:{len(builders) + 1:04d}"
                current = _SectionBuilder(
                    section_id=section_id,
                    index=index,
                    title=heading_title,
                    level=level,
                    parent_section_id=parent_id,
                    heading_page=page_number,
                )
                builders.append(current)
                level_stack[level] = section_id
                for deeper_level in [item for item in level_stack if item > level]:
                    del level_stack[deeper_level]
            flush(page_number)

        if not builders:
            blocks = [
                PageTextBlock(
                    page_number=page_number,
                    text=normalized,
                )
                for page_number, text in enumerate(page_texts, start=1)
                if (normalized := self._normalize_body_lines(text.splitlines()))
            ]
            return [
                PaperSection(
                    section_id=f"{paper_id}:section:0001",
                    title="Document Body",
                    level=1,
                    blocks=blocks,
                    page_start=1,
                    page_end=len(page_texts),
                )
            ]
        return [builder.build() for builder in builders]

    def _parse_heading(
        self,
        line: str,
        *,
        allow_named: bool = True,
    ) -> tuple[str | None, str, int] | None:
        if not line:
            return None
        numbered = self._valid_numbered_heading(line)
        if numbered is not None:
            return numbered
        normalized = line.casefold().rstrip(":")
        always_named = {"abstract", "references", "limitations", "acknowledgments"}
        if normalized in self._NAMED_HEADINGS and (
            allow_named or normalized in always_named
        ):
            return None, line.rstrip(":"), 1
        return None

    def _valid_numbered_heading(self, line: str) -> tuple[str, str, int] | None:
        numbered = self._NUMBERED_HEADING.fullmatch(line)
        if numbered is None:
            return None
        title = numbered.group("title").strip()
        index = numbered.group("index")
        if "." not in index and index.isdigit() and int(index) > 9:
            return None
        if not self._looks_like_heading_title(title):
            return None
        return index, title, index.count(".") + 1

    @staticmethod
    def _looks_like_heading_title(title: str) -> bool:
        forbidden = {"←", "→", "∅", "∪", "=", "≤", "≥"}
        if (
            len(title.split()) > 16
            or title.endswith((".", "?", "!"))
            or any(symbol in title for symbol in forbidden)
            or not title[:1].isupper()
        ):
            return False
        return any(character.isalpha() for character in title)

    @staticmethod
    def _normalize_body_lines(lines: list[str]) -> str:
        paragraphs: list[str] = []
        current: list[str] = []
        for raw_line in lines:
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            if current and current[-1].endswith("-") and line[:1].islower():
                current[-1] = current[-1][:-1] + line
            else:
                current.append(line)
        if current:
            paragraphs.append(" ".join(current))
        return "\n\n".join(paragraphs).strip()

    @classmethod
    def _detect_title(cls, first_page_text: str) -> str | None:
        lines = [re.sub(r"\s+", " ", item).strip() for item in first_page_text.splitlines()[:25]]
        candidates: list[str] = []
        for index, line in enumerate(lines):
            if line.casefold() == "abstract":
                break
            if cls._is_title_noise(line):
                if candidates:
                    break
                continue
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            if candidates and cls._looks_like_author_line(line, next_line):
                break
            if 8 <= len(line) <= 240 and any(character.isalpha() for character in line):
                candidates.append(line)
                if len(candidates) == 3:
                    break
        return " ".join(candidates) if candidates else None

    @classmethod
    def _extract_metadata(
        cls,
        page_texts: list[str],
        sections: list[PaperSection],
        pdf_metadata: Mapping[object, object],
        *,
        supplied_title: str | None,
        fallback_title: str,
        identifier_text: str,
    ) -> PaperMetadata:
        embedded_title = cls._clean_metadata_value(pdf_metadata.get("/Title"))
        title = (
            (supplied_title or "").strip()
            or cls._detect_title(page_texts[0])
            or embedded_title
            or fallback_title
        )
        embedded_authors = cls._clean_metadata_value(pdf_metadata.get("/Author"))
        authors = cls._split_authors(embedded_authors) if embedded_authors else []
        if not authors:
            authors = cls._detect_authors(page_texts[0], title)
        abstract = next(
            (
                "\n\n".join(block.text for block in section.blocks).strip()
                for section in sections
                if section.title.casefold().strip() == "abstract"
            ),
            None,
        )
        searchable_text = "\n".join([*page_texts[:2], identifier_text])
        embedded_keywords = cls._clean_metadata_value(pdf_metadata.get("/Keywords"))
        keywords = cls._extract_keywords(searchable_text, embedded_keywords)
        doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", searchable_text, re.I)
        arxiv_match = re.search(r"\barXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", searchable_text, re.I)
        return PaperMetadata(
            title=title,
            authors=authors,
            abstract=abstract or None,
            keywords=keywords,
            doi=doi_match.group(0).rstrip(".,;) ") if doi_match else None,
            arxiv_id=arxiv_match.group(1) if arxiv_match else None,
        )

    @classmethod
    def _detect_authors(cls, first_page_text: str, title: str) -> list[str]:
        raw_lines = first_page_text.splitlines()[:50]
        lines = [re.sub(r"\s+", " ", item).strip() for item in raw_lines]
        abstract_index = next(
            (index for index, line in enumerate(lines) if line.casefold() == "abstract"),
            len(lines),
        )
        normalized_title = re.sub(r"\s+", " ", title).casefold()
        title_end = 0
        assembled = ""
        for index, line in enumerate(lines[:abstract_index]):
            if not line or cls._is_title_noise(line):
                continue
            assembled = f"{assembled} {line}".strip()
            assembled_normalized = assembled.casefold()
            if (
                assembled_normalized in normalized_title
                or normalized_title.startswith(assembled_normalized)
            ):
                title_end = index + 1
                if assembled_normalized == normalized_title:
                    break
        authors: list[str] = []
        for index in range(title_end, abstract_index):
            line = lines[index]
            next_line = lines[index + 1] if index + 1 < abstract_index else ""
            column_parts = [
                part.strip()
                for part in re.split(r"\s{2,}", raw_lines[index].strip())
                if part.strip()
            ]
            candidates = column_parts if len(column_parts) > 1 else [line]
            for candidate in candidates:
                if cls._looks_like_author_line(candidate, next_line):
                    authors.extend(cls._split_authors(candidate))
        return list(dict.fromkeys(authors))

    @classmethod
    def _looks_like_author_line(cls, line: str, next_line: str = "") -> bool:
        lowered = line.casefold()
        if (
            not line
            or "@" in line
            or any(
                marker in lowered
                for marker in (
                    "university", "institute", "school", "laboratory",
                    "college", "department", "proceedings", "association",
                    "future house", "align to innovate", "research institute",
                )
            )
        ):
            return False
        lowered_next = next_line.casefold()
        affiliation_markers = (
            "university", "institute", "school", "laboratory", "college",
            "research", "future house", "align to innovate", "department",
        )
        parts = [part.strip() for part in re.split(r",|\band\b", line) if part.strip()]
        cleaned = [re.sub(r"[\d*†‡∗]+$", "", part).strip() for part in parts]
        names = [part for part in cleaned if cls._looks_like_person_name(part)]
        return bool(names) and (
            len(names) >= 2 and len(parts) > 1
            or any(marker in lowered_next for marker in affiliation_markers)
        )

    @staticmethod
    def _looks_like_person_name(value: str) -> bool:
        words = value.split()
        if not 2 <= len(words) <= 6 or any(char in value for char in "@{}"):
            return False
        return all(any(character.isalpha() for character in word) for word in words)

    @classmethod
    def _split_authors(cls, value: str) -> list[str]:
        candidates = re.split(r"\s*(?:,|;|\band\b)\s*", value)
        return [
            cleaned
            for candidate in candidates
            if (cleaned := re.sub(r"[\d*†‡∗]+$", "", candidate).strip())
            and cls._looks_like_person_name(cleaned)
        ]

    @staticmethod
    def _clean_metadata_value(value: object) -> str | None:
        cleaned = re.sub(r"\s+", " ", str(value or "")).strip()
        return cleaned or None

    @staticmethod
    def _extract_keywords(text: str, embedded_keywords: str | None) -> list[str]:
        value = embedded_keywords
        if not value:
            match = re.search(
                r"^(?:keywords|index terms)\s*[-—:]\s*(.+)$",
                text,
                re.I | re.M,
            )
            value = match.group(1) if match else None
        if not value:
            return []
        return list(
            dict.fromkeys(
                item.strip().rstrip(".")
                for item in re.split(r"[,;]", value)
                if item.strip()
            )
        )

    @staticmethod
    def _is_title_noise(line: str) -> bool:
        lowered = line.casefold()
        venue_markers = (
            "proceedings of",
            "association for computational linguistics",
            "copyright",
            "©",
            " pages ",
        )
        author_or_affiliation = (
            "@" in line
            or "university" in lowered
            or "institute" in lowered
            or "school of" in lowered
            or "laboratory" in lowered
            or (line.count(",") >= 2 and any(character.isdigit() for character in line))
        )
        date_line = bool(
            re.search(
                r"\b(?:january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\b.*\b(?:19|20)\d{2}\b",
                lowered,
            )
        )
        return (
            not line
            or any(marker in lowered for marker in venue_markers)
            or author_or_affiliation
            or date_line
            or bool(re.fullmatch(r"[\d\W]+", line))
        )
