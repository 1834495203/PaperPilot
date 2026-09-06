import asyncio
import importlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pypdf import PageObject, PdfReader
from pypdf.errors import PdfReadError

from app.domain.ports import ScientificPaperParser
from app.domain.rag import (
    BoundingBox,
    CaptionBlock,
    CodeBlock,
    DocumentBlock,
    EquationBlock,
    FigureBlock,
    PageTextBlock,
    PaperBlockType,
    PaperMetadata,
    PaperSection,
    ParsedPaperDocument,
    TableBlock,
    TextSpan,
)


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
    semantic_role: str | None = None
    blocks: list[DocumentBlock] = field(default_factory=list)

    def build(self) -> PaperSection:
        page_end = max(
            (block.page_number for block in self.blocks),
            default=self.heading_page,
        )
        return PaperSection(
            section_id=self.section_id,
            index=self.index,
            title=self.title,
            semantic_role=self.semantic_role,
            level=self.level,
            parent_section_id=self.parent_section_id,
            blocks=self.blocks,
            page_start=self.heading_page,
            page_end=page_end,
        )


@dataclass(frozen=True, slots=True)
class _LayoutLine:
    page_number: int
    page_width: float
    page_height: float
    text: str
    bbox: BoundingBox
    spans: tuple[TextSpan, ...]
    font_size: float
    bold: bool
    italic: bool
    source_block: int = -1
    gap_before: float = 0.0

    @property
    def style_key(self) -> tuple[float, bool, bool, int]:
        return (round(self.font_size, 1), self.bold, self.italic, round(self.bbox.x0 / 12))


@dataclass(frozen=True, slots=True)
class _LayoutPage:
    page_number: int
    width: float
    height: float
    lines: tuple[_LayoutLine, ...]
    tables: tuple[TableBlock, ...]
    figures: tuple[FigureBlock, ...]
    code_blocks: tuple[CodeBlock, ...] = ()


class PypdfScientificPaperParser(ScientificPaperParser):
    """Extract page-aware scientific-paper sections while preserving native headings."""

    _NUMBERED_HEADING = re.compile(
        r"^(?P<index>(?:\d+(?:\.\d+){0,4}|[A-H](?:\.\d+){0,3}))"
        r"[.)]?\s+(?P<title>\S.{1,159})$"
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
        asset_dir: Path | None = None,
    ) -> ParsedPaperDocument:
        return await asyncio.to_thread(
            self._parse_sync,
            Path(path),
            paper_id,
            title,
            asset_dir,
        )

    def _parse_sync(
        self,
        path: Path,
        paper_id: str,
        title: str | None,
        asset_dir: Path | None,
    ) -> ParsedPaperDocument:
        if path.suffix.lower() != ".pdf":
            raise ScientificPdfParseError("Only PDF documents can be ingested")
        if not path.is_file():
            raise ScientificPdfParseError(f"PDF does not exist: {path}")
        try:
            reader = PdfReader(path, strict=False)
            if reader.is_encrypted and reader.decrypt("") == 0:
                raise ScientificPdfParseError("Encrypted PDF requires a password")
            layout_pages = self._extract_layout_pages(path, asset_dir)
            page_texts = ["\n".join(line.text for line in page.lines) for page in layout_pages]
            first_page_plain_text = str(reader.pages[0].extract_text() or "")
            pdf_metadata = dict(reader.metadata or {})
        except (PdfReadError, OSError, RuntimeError, ValueError) as error:
            raise ScientificPdfParseError(f"Unable to parse PDF: {error}") from error
        if not any(text.strip() for text in page_texts):
            raise ScientificPdfParseError("PDF contains no extractable text; OCR is not supported")

        repeated_lines = self._repeated_page_lines(page_texts)
        sections = self._extract_layout_sections(paper_id, layout_pages, repeated_lines)
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
            page_count=len(layout_pages),
            sections=sections,
            page_dimensions={
                page.page_number: (page.width, page.height) for page in layout_pages
            },
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

    def _extract_layout_pages(
        self,
        path: Path,
        asset_dir: Path | None,
    ) -> list[_LayoutPage]:
        try:
            pymupdf = importlib.import_module("pymupdf")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "PyMuPDF is required for layout-aware scientific PDF parsing"
            ) from error
        pages: list[_LayoutPage] = []
        seen_xrefs: set[int] = set()
        with pymupdf.open(str(path)) as document:
            for page_index, page in enumerate(document, start=1):
                page_rect = page.rect
                width = float(page_rect.width)
                height = float(page_rect.height)
                raw = cast(dict[str, Any], page.get_text("dict"))
                raw_images = self._extract_raw_images(
                    document, page, page_index, asset_dir, seen_xrefs
                )
                lines: list[_LayoutLine] = []
                figures: list[FigureBlock] = []
                figure_index = 0
                for block_index, block in enumerate(raw.get("blocks", [])):
                    if int(block.get("type", 0)) == 1:
                        bbox = self._bbox(block.get("bbox"))
                        if bbox is not None and self._bbox_area(bbox) >= width * height * 0.01:
                            asset_ref = self._save_figure_png(
                                page, pymupdf, bbox, page_index, figure_index, asset_dir
                            )
                            figure_index += 1
                            figures.append(
                                FigureBlock(
                                    page_number=page_index,
                                    bbox=bbox,
                                    parse_confidence=0.7,
                                    text=f"Figure region on page {page_index}",
                                    asset_ref=asset_ref,
                                    raw_asset_ref=self._match_raw_image(bbox, raw_images),
                                )
                            )
                        continue
                    for raw_line in block.get("lines", []):
                        spans: list[TextSpan] = []
                        for raw_span in raw_line.get("spans", []):
                            text = str(raw_span.get("text", ""))
                            bbox = self._bbox(raw_span.get("bbox"))
                            if not text or bbox is None:
                                continue
                            font_name = str(raw_span.get("font", "")) or None
                            flags = int(raw_span.get("flags", 0))
                            lowered_font = (font_name or "").casefold()
                            spans.append(
                                TextSpan(
                                    text=text,
                                    bbox=bbox,
                                    font_name=font_name,
                                    font_size=max(float(raw_span.get("size", 0.0)), 0.1),
                                    bold="bold" in lowered_font or bool(flags & 16),
                                    italic=(
                                        "italic" in lowered_font
                                        or "oblique" in lowered_font
                                        or bool(flags & 2)
                                    ),
                                )
                            )
                        if not spans:
                            continue
                        text = "".join(span.text for span in spans).strip()
                        line_bbox = self._bbox(raw_line.get("bbox"))
                        if not text or line_bbox is None:
                            continue
                        lines.append(
                            _LayoutLine(
                                page_number=page_index,
                                page_width=width,
                                page_height=height,
                                text=text,
                                bbox=line_bbox,
                                spans=tuple(spans),
                                font_size=self._dominant_font_size(spans),
                                bold=self._style_ratio(spans, "bold") >= 0.8,
                                italic=self._style_ratio(spans, "italic") >= 0.8,
                                source_block=block_index,
                            )
                        )
                merged_lines = self._merge_adjacent_lines(lines)
                merged_lines = self._merge_wrapped_heading_lines(merged_lines)
                merged_lines = self._merge_caption_continuations(merged_lines)
                ordered_lines = self._reading_order(merged_lines, width)
                code_blocks, ordered_lines = self._extract_code_blocks(
                    page, page_index, ordered_lines
                )
                tables = self._extract_tables(page, page_index, merged_lines)
                pages.append(
                    _LayoutPage(
                        page_number=page_index,
                        width=width,
                        height=height,
                        lines=tuple(self._with_vertical_gaps(ordered_lines)),
                        tables=tuple(tables),
                        figures=tuple(figures),
                        code_blocks=tuple(code_blocks),
                    )
                )
        return pages

    @staticmethod
    def _save_figure_png(
        page: Any,
        pymupdf: Any,
        bbox: BoundingBox,
        page_number: int,
        figure_index: int,
        asset_dir: Path | None,
    ) -> str | None:
        """Render a detected figure region to a PNG inside the paper's asset directory."""
        if asset_dir is None:
            return None
        filename = f"fig-{page_number:04d}-{figure_index:02d}.png"
        try:
            clip = pymupdf.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            pixmap = page.get_pixmap(clip=clip, dpi=144)
            asset_dir.mkdir(parents=True, exist_ok=True)
            pixmap.save(asset_dir / filename)
        except Exception:
            return None
        return filename

    @staticmethod
    def _extract_raw_images(
        document: Any,
        page: Any,
        page_number: int,
        asset_dir: Path | None,
        seen_xrefs: set[int],
    ) -> list[tuple[BoundingBox, str]]:
        """Save embedded raster images and return their placement rects + filenames."""
        if asset_dir is None:
            return []
        page_area = float(page.rect.width * page.rect.height)
        results: list[tuple[BoundingBox, str]] = []
        for info in page.get_images(full=True):
            xref = int(info[0])
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue
            significant: list[BoundingBox] = []
            for rect in rects:
                bbox = PypdfScientificPaperParser._bbox(rect)
                if (
                    bbox is not None
                    and PypdfScientificPaperParser._bbox_area(bbox) >= page_area * 0.01
                ):
                    significant.append(bbox)
            if not significant:
                continue
            try:
                meta = document.extract_image(xref)
                extension = str(meta.get("ext") or "png")
                data = meta.get("image")
            except Exception:
                continue
            if not data:
                continue
            filename = f"raw-{page_number:04d}-{xref}.{extension}"
            try:
                asset_dir.mkdir(parents=True, exist_ok=True)
                (asset_dir / filename).write_bytes(data)
            except OSError:
                continue
            for bbox in significant:
                results.append((bbox, filename))
        return results

    @staticmethod
    def _match_raw_image(
        bbox: BoundingBox,
        raw_images: list[tuple[BoundingBox, str]],
    ) -> str | None:
        best: str | None = None
        best_area = 0.0
        for image_bbox, filename in raw_images:
            intersection = PypdfScientificPaperParser._intersection_area(bbox, image_bbox)
            if intersection > best_area:
                best_area = intersection
                best = filename
        return best if best_area > 0 else None

    @staticmethod
    def _intersection_area(left: BoundingBox, right: BoundingBox) -> float:
        x_overlap = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
        y_overlap = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
        return x_overlap * y_overlap

    def _extract_code_blocks(
        self,
        page: Any,
        page_number: int,
        lines: list[_LayoutLine],
    ) -> tuple[list[CodeBlock], list[_LayoutLine]]:
        """Detect boxed algorithm/pseudocode regions and merge them into atomic blocks."""
        boxes = self._merge_adjacent_boxes(self._rule_boxes(page))
        if not boxes or not lines:
            return [], lines

        code_boxes: list[tuple[float, float, float, float]] = []
        for box in boxes:
            inside = [line for line in lines if self._line_in_box(line, box)]
            if not inside:
                continue
            titled = any(
                re.match(
                    r"^(algorithm|procedure|function)\b",
                    line.text.strip(),
                    re.IGNORECASE,
                )
                for line in inside
            )
            if titled:
                code_boxes.append(box)

        if not code_boxes:
            return [], lines

        code_blocks: list[CodeBlock] = []
        removed: set[int] = set()
        for box in code_boxes:
            inside = [
                (index, line)
                for index, line in enumerate(lines)
                if self._line_in_box(line, box)
            ]
            if not inside:
                continue
            text = "\n".join(line.text for _, line in inside).strip()
            code_blocks.append(
                CodeBlock(
                    page_number=page_number,
                    bbox=BoundingBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
                    parse_confidence=0.85,
                    text=text,
                )
            )
            removed.update(index for index, _ in inside)
        remaining = [line for index, line in enumerate(lines) if index not in removed]
        return code_blocks, remaining

    @staticmethod
    def _merge_adjacent_boxes(
        boxes: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        """Merge vertically-adjacent boxes that share the same horizontal extent."""
        if not boxes:
            return []
        merged: list[tuple[float, float, float, float]] = []
        for box in sorted(boxes, key=lambda item: (item[1], item[0])):
            if merged:
                previous = merged[-1]
                same_extent = (
                    abs(previous[0] - box[0]) <= 8 and abs(previous[2] - box[2]) <= 8
                )
                contiguous = -5 <= box[1] - previous[3] <= 5
                if same_extent and contiguous:
                    merged[-1] = (previous[0], previous[1], previous[2], box[3])
                    continue
            merged.append(box)
        return merged


    @staticmethod
    def _rule_boxes(page: Any) -> list[tuple[float, float, float, float]]:
        """Pair horizontal vector rules into (x0, y0, x1, y1) rectangles."""
        get_drawings = getattr(page, "get_drawings", None)
        if get_drawings is None:
            return []
        page_width = float(page.rect.width)
        rules: list[tuple[float, float, float]] = []
        try:
            drawings = get_drawings()
        except Exception:
            return []
        for drawing in drawings:
            rect = drawing.get("rect")
            if rect is None:
                continue
            if float(rect.height) <= 1.5 and float(rect.width) >= page_width * 0.3:
                rules.append((float(rect.y0), float(rect.x0), float(rect.x1)))
        rules.sort()
        boxes: list[tuple[float, float, float, float]] = []
        for index, (y0, x0, x1) in enumerate(rules):
            for y1, bottom_x0, bottom_x1 in rules[index + 1 :]:
                if y1 - y0 < 20:
                    continue
                if abs(x0 - bottom_x0) <= 8 and abs(x1 - bottom_x1) <= 8:
                    boxes.append((min(x0, bottom_x0), y0, max(x1, bottom_x1), y1))
                    break
        return boxes

    @staticmethod
    def _line_in_box(
        line: _LayoutLine,
        box: tuple[float, float, float, float],
    ) -> bool:
        x0, y0, x1, y1 = box
        pad = 2.0
        return (
            line.bbox.x0 >= x0 - pad
            and line.bbox.x1 <= x1 + pad
            and line.bbox.y0 >= y0 - pad
            and line.bbox.y1 <= y1 + pad
        )

    @classmethod
    def _extract_tables(
        cls,
        page: Any,
        page_number: int,
        lines: list[_LayoutLine],
    ) -> list[TableBlock]:
        find_tables = getattr(page, "find_tables", None)
        if find_tables is None:
            return []
        try:
            detected = find_tables()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return []
        tables = list(getattr(detected, "tables", []))
        used_text_fallback = False
        table_captions = [
            line
            for line in lines
            if (parts := cls._caption_parts(line.text)) is not None
            and parts[0] is PaperBlockType.TABLE
        ]
        if not tables and table_captions:
            try:
                detected = find_tables(
                    strategy="text",
                    min_words_vertical=2,
                    min_words_horizontal=1,
                )
                tables = list(getattr(detected, "tables", []))
                used_text_fallback = True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                tables = []
        blocks: list[TableBlock] = []
        for table in tables:
            bbox = cls._bbox(getattr(table, "bbox", None))
            if bbox is None:
                continue
            if used_text_fallback:
                page_area = lines[0].page_width * lines[0].page_height
                if cls._bbox_area(bbox) > page_area * 0.45:
                    continue
            if table_captions:
                caption_distance = min(
                    min(abs(line.bbox.y0 - bbox.y1), abs(bbox.y0 - line.bbox.y1))
                    for line in table_captions
                )
                if caption_distance > lines[0].page_height * 0.35:
                    continue
            try:
                extracted = table.extract()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                extracted = []
            rows = [
                [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
                for row in extracted
                if row
            ]
            if not any(any(cell for cell in row) for row in rows):
                continue
            markdown = cls._table_markdown(rows)
            blocks.append(
                TableBlock(
                    page_number=page_number,
                    bbox=bbox,
                    parse_confidence=0.85,
                    text=markdown,
                    rows=rows,
                    markdown=markdown,
                )
            )
        if table_captions and not blocks:
            blocks.extend(
                TableBlock(
                    page_number=page_number,
                    bbox=line.bbox,
                    parse_confidence=0.45,
                    text=line.text,
                    object_label=cls._caption_parts(line.text)[1],  # type: ignore[index]
                    caption=line.text,
                )
                for line in table_captions
            )
        return blocks

    @staticmethod
    def _bbox(value: object) -> BoundingBox | None:
        if value is None:
            return None
        if all(hasattr(value, attribute) for attribute in ("x0", "y0", "x1", "y1")):
            rect = cast(Any, value)
            return BoundingBox(
                x0=float(rect.x0),
                y0=float(rect.y0),
                x1=float(rect.x1),
                y1=float(rect.y1),
            )
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return None
        coordinates = list(value)
        if len(coordinates) != 4:
            return None
        return BoundingBox(
            x0=float(coordinates[0]),
            y0=float(coordinates[1]),
            x1=float(coordinates[2]),
            y1=float(coordinates[3]),
        )

    @staticmethod
    def _bbox_area(bbox: BoundingBox) -> float:
        return max(0.0, bbox.x1 - bbox.x0) * max(0.0, bbox.y1 - bbox.y0)

    @staticmethod
    def _dominant_font_size(spans: list[TextSpan]) -> float:
        weights: Counter[float] = Counter()
        for span in spans:
            if span.font_size is not None:
                weights[round(span.font_size, 1)] += max(len(span.text.strip()), 1)
        return weights.most_common(1)[0][0] if weights else 10.0

    @staticmethod
    def _style_ratio(spans: list[TextSpan], attribute: str) -> float:
        visible = [span for span in spans if span.text.strip()]
        total = sum(len(span.text.strip()) for span in visible)
        if total == 0:
            return 0.0
        styled = sum(
            len(span.text.strip()) for span in visible if bool(getattr(span, attribute))
        )
        return styled / total

    @staticmethod
    def _merge_adjacent_lines(lines: list[_LayoutLine]) -> list[_LayoutLine]:
        """Join heading fragments such as a detached section number and title."""
        ordered = list(lines)
        merged: list[_LayoutLine] = []
        for line in ordered:
            previous = merged[-1] if merged else None
            if previous is None:
                merged.append(line)
                continue
            horizontal_gap = line.bbox.x0 - previous.bbox.x1
            same_baseline = abs(line.bbox.y0 - previous.bbox.y0) <= 2.5
            close_enough = 0 <= horizontal_gap <= max(30.0, line.font_size * 3)
            number_prefix = re.fullmatch(
                r"(?:\d+(?:\.\d+){0,4}|[A-H](?:\.\d+){0,3})[.)]?",
                previous.text.strip(),
            )
            similar_font = (
                abs(line.font_size - previous.font_size) <= 1.0
                or number_prefix is not None
            )
            same_source = line.source_block == previous.source_block
            if not (same_source and same_baseline and close_enough and similar_font):
                merged.append(line)
                continue
            separator = (
                " "
                if previous.text[-1:].isalnum() and line.text[:1].isalnum()
                else ""
            )
            merged[-1] = _LayoutLine(
                page_number=line.page_number,
                page_width=line.page_width,
                page_height=line.page_height,
                text=f"{previous.text}{separator}{line.text}",
                bbox=BoundingBox(
                    x0=previous.bbox.x0,
                    y0=min(previous.bbox.y0, line.bbox.y0),
                    x1=line.bbox.x1,
                    y1=max(previous.bbox.y1, line.bbox.y1),
                ),
                spans=(*previous.spans, *line.spans),
                font_size=max(previous.font_size, line.font_size),
                bold=previous.bold or line.bold,
                italic=previous.italic or line.italic,
                source_block=line.source_block,
            )
        return merged

    def _merge_wrapped_heading_lines(
        self,
        lines: list[_LayoutLine],
    ) -> list[_LayoutLine]:
        merged: list[_LayoutLine] = []
        for line in lines:
            previous = merged[-1] if merged else None
            numbered = self._valid_numbered_heading(previous.text) if previous else None
            if (
                previous is None
                or numbered is None
                or line.source_block != previous.source_block
                or abs(line.font_size - previous.font_size) > 1.0
                or line.bold != previous.bold
                or line.bbox.y0 - previous.bbox.y1 > line.font_size * 0.9
                or len(line.text.split()) > 8
            ):
                merged.append(line)
                continue
            merged[-1] = _LayoutLine(
                page_number=previous.page_number,
                page_width=previous.page_width,
                page_height=previous.page_height,
                text=f"{previous.text} {line.text}",
                bbox=BoundingBox(
                    x0=min(previous.bbox.x0, line.bbox.x0),
                    y0=previous.bbox.y0,
                    x1=max(previous.bbox.x1, line.bbox.x1),
                    y1=line.bbox.y1,
                ),
                spans=(*previous.spans, *line.spans),
                font_size=previous.font_size,
                bold=previous.bold,
                italic=previous.italic,
                source_block=previous.source_block,
            )
        return merged

    def _merge_caption_continuations(
        self,
        lines: list[_LayoutLine],
    ) -> list[_LayoutLine]:
        """Fold wrapped caption lines into their caption's first line so captions stay intact."""
        merged: list[_LayoutLine] = []
        index = 0
        while index < len(lines):
            caption = lines[index]
            merged.append(caption)
            index += 1
            if self._caption_parts(caption.text) is None:
                continue
            while index < len(lines) and self._is_caption_continuation(caption, lines[index]):
                next_line = lines[index]
                caption = _LayoutLine(
                    page_number=caption.page_number,
                    page_width=caption.page_width,
                    page_height=caption.page_height,
                    text=f"{caption.text} {next_line.text}",
                    bbox=BoundingBox(
                        x0=min(caption.bbox.x0, next_line.bbox.x0),
                        y0=caption.bbox.y0,
                        x1=max(caption.bbox.x1, next_line.bbox.x1),
                        y1=next_line.bbox.y1,
                    ),
                    spans=(*caption.spans, *next_line.spans),
                    font_size=max(caption.font_size, next_line.font_size),
                    bold=caption.bold or next_line.bold,
                    italic=caption.italic or next_line.italic,
                    source_block=next_line.source_block,
                )
                merged[-1] = caption
                index += 1
        return merged

    def _is_caption_continuation(self, caption: _LayoutLine, line: _LayoutLine) -> bool:
        if self._caption_parts(line.text) is not None:
            return False
        if self._looks_like_equation(line.text):
            return False
        if self._valid_numbered_heading(line.text) is not None:
            return False
        if abs(line.bbox.x0 - caption.bbox.x0) > 12:
            return False
        if abs(line.font_size - caption.font_size) > 1.5:
            return False
        gap = line.bbox.y0 - caption.bbox.y1
        return -line.font_size <= gap <= line.font_size * 1.5

    @classmethod
    def _reading_order(cls, lines: list[_LayoutLine], page_width: float) -> list[_LayoutLine]:
        if not lines:
            return []
        mid = page_width / 2
        narrow = [line for line in lines if line.bbox.x1 - line.bbox.x0 < page_width * 0.62]
        left = [line for line in narrow if line.bbox.x1 <= mid * 1.08]
        right = [line for line in narrow if line.bbox.x0 >= mid * 0.92]
        two_columns = len(left) >= 4 and len(right) >= 4
        if not two_columns:
            return sorted(lines, key=lambda item: (item.bbox.y0, item.bbox.x0))

        wide = sorted(
            [line for line in lines if line not in left and line not in right],
            key=lambda item: (item.bbox.y0, item.bbox.x0),
        )
        column_lines = [line for line in lines if line in left or line in right]
        ordered: list[_LayoutLine] = []
        lower_bound = float("-inf")
        for separator in [*wide, None]:
            upper_bound = separator.bbox.y0 if separator is not None else float("inf")
            band = [
                line
                for line in column_lines
                if lower_bound <= line.bbox.y0 < upper_bound
            ]
            ordered.extend(
                sorted((line for line in band if line in left), key=lambda x: x.bbox.y0)
            )
            ordered.extend(
                sorted((line for line in band if line in right), key=lambda x: x.bbox.y0)
            )
            if separator is not None:
                ordered.append(separator)
                lower_bound = separator.bbox.y1
        return ordered

    @staticmethod
    def _with_vertical_gaps(lines: list[_LayoutLine]) -> list[_LayoutLine]:
        result: list[_LayoutLine] = []
        previous_by_column: dict[int, _LayoutLine] = {}
        for line in lines:
            column = 0 if line.bbox.x0 < line.page_width / 2 else 1
            previous = previous_by_column.get(column)
            gap = max(0.0, line.bbox.y0 - previous.bbox.y1) if previous else 0.0
            result.append(
                _LayoutLine(
                    page_number=line.page_number,
                    page_width=line.page_width,
                    page_height=line.page_height,
                    text=line.text,
                    bbox=line.bbox,
                    spans=line.spans,
                    font_size=line.font_size,
                    bold=line.bold,
                    italic=line.italic,
                    source_block=line.source_block,
                    gap_before=gap,
                )
            )
            previous_by_column[column] = line
        return result

    @staticmethod
    def _table_markdown(rows: list[list[str]]) -> str:
        width = max((len(row) for row in rows), default=0)
        if width == 0:
            return ""
        normalized = [row + [""] * (width - len(row)) for row in rows]

        def render(row: list[str]) -> str:
            return "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"

        rendered_rows = [render(row) for row in normalized[1:]]
        return "\n".join(
            [render(normalized[0]), render(["---"] * width), *rendered_rows]
        )

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

    def _extract_layout_sections(
        self,
        paper_id: str,
        pages: list[_LayoutPage],
        repeated_lines: set[str],
    ) -> list[PaperSection]:
        usable_lines = [
            line
            for page in pages
            for line in page.lines
            if line.text.strip() and line.text.strip() not in repeated_lines
        ]
        if not usable_lines:
            return []
        body_font_size = self._infer_body_font_size(usable_lines)
        style_counts = Counter(line.style_key for line in usable_lines)
        candidates = [
            line
            for line in usable_lines
            if self._is_heading_candidate(line, body_font_size, style_counts)
        ]
        candidates = self._filter_heading_candidates(candidates)
        heading_levels = self._infer_heading_levels(candidates, body_font_size)
        heading_keys = {self._line_key(line): line for line in candidates}

        builders: list[_SectionBuilder] = []
        current: _SectionBuilder | None = None
        level_stack: dict[int, str] = {}

        def ensure_front_matter(page_number: int) -> _SectionBuilder:
            nonlocal current
            if current is None:
                current = _SectionBuilder(
                    section_id=f"{paper_id}:section:{len(builders) + 1:04d}",
                    index=None,
                    title="Front Matter",
                    semantic_role="front_matter",
                    level=1,
                    parent_section_id=None,
                    heading_page=page_number,
                )
                builders.append(current)
            return current

        for page in pages:
            page_objects = self._linked_page_objects(page)
            object_index = 0
            for line in page.lines:
                while (
                    object_index < len(page_objects)
                    and self._block_y(page_objects[object_index]) <= line.bbox.y0
                ):
                    ensure_front_matter(page.page_number).blocks.append(
                        page_objects[object_index]
                    )
                    object_index += 1
                if (
                    line.text.strip() in repeated_lines
                    or self._is_page_number_line(line)
                    or self._inside_objects(line, page_objects)
                ):
                    continue
                key = self._line_key(line)
                if key in heading_keys:
                    index, title = self._heading_parts(line.text)
                    level = heading_levels[key]
                    parent_id = next(
                        (
                            level_stack[item]
                            for item in range(level - 1, 0, -1)
                            if item in level_stack
                        ),
                        None,
                    )
                    current = _SectionBuilder(
                        section_id=f"{paper_id}:section:{len(builders) + 1:04d}",
                        index=index,
                        title=title,
                        semantic_role=self._semantic_role(title),
                        level=level,
                        parent_section_id=parent_id,
                        heading_page=page.page_number,
                    )
                    builders.append(current)
                    level_stack[level] = current.section_id
                    for deeper in [item for item in level_stack if item > level]:
                        del level_stack[deeper]
                    continue
                active = ensure_front_matter(page.page_number)
                active.blocks.append(
                    self._line_block(paper_id, line, len(active.blocks))
                )
            while object_index < len(page_objects):
                ensure_front_matter(page.page_number).blocks.append(page_objects[object_index])
                object_index += 1

        sections = [builder.build() for builder in builders if builder.blocks or builder.title]
        return [
            section.model_copy(update={"blocks": self._merge_text_blocks(section.blocks)})
            for section in sections
        ]

    @staticmethod
    def _infer_body_font_size(lines: list[_LayoutLine]) -> float:
        weights: Counter[float] = Counter()
        for line in lines:
            if not 0.04 * line.page_height <= line.bbox.y0 <= 0.96 * line.page_height:
                continue
            for span in line.spans:
                if span.font_size is not None:
                    weights[round(span.font_size, 1)] += len(span.text.strip())
        return weights.most_common(1)[0][0] if weights else 10.0

    def _is_heading_candidate(
        self,
        line: _LayoutLine,
        body_font_size: float,
        style_counts: Counter[tuple[float, bool, bool, int]],
    ) -> bool:
        text = re.sub(r"\s+", " ", line.text).strip()
        if line.font_size < body_font_size * 0.8:
            return False
        if (
            len(text) < 3
            or len(text) > 180
            or len(text.split()) > 22
            or not any(character.isalpha() for character in text)
            or self._caption_parts(text) is not None
            or self._looks_like_equation(text)
            or text.endswith((".", "!", ";"))
            or (". " in text and len(text.split()) > 8)
            or text.endswith("-")
        ):
            return False
        numbered = self._valid_numbered_heading(text) is not None
        lexical_hint = text.casefold().rstrip(":") in self._NAMED_HEADINGS
        if not numbered and not lexical_hint and len(text) > 75:
            return False
        ratio = line.font_size / max(body_font_size, 0.1)
        letters = "".join(character for character in text if character.isalpha())
        uppercase_title = bool(letters) and letters.upper() == letters
        if numbered and ratio < 1.02 and not line.bold and not uppercase_title:
            return False
        score = 0.0
        score += 3.0 if ratio >= 1.14 else 1.0 if ratio >= 1.04 else 0.0
        score += 2.0 if line.bold else 0.0
        score += 3.0 if numbered else 0.0
        score += 3.0 if lexical_hint else 0.0
        score += 2.0 if line.gap_before >= body_font_size * 0.8 else 0.0
        score += 1.0 if 2 <= style_counts[line.style_key] <= 80 else 0.0
        score += 1.0 if len(text) <= 100 else 0.0
        return score >= 5.0

    def _filter_heading_candidates(
        self,
        candidates: list[_LayoutLine],
    ) -> list[_LayoutLine]:
        filtered = [line for line in candidates if not self._looks_like_running_matter(line)]
        anchor_index = next(
            (
                index
                for index, line in enumerate(filtered)
                if self._valid_numbered_heading(line.text) is not None
                or self._semantic_role(line.text) is not None
            ),
            None,
        )
        if anchor_index is None:
            return filtered
        anchor = filtered[anchor_index]
        if anchor.page_number == 1:
            return filtered[anchor_index:]
        return [line for line in filtered if line.page_number > 1]

    @staticmethod
    def _looks_like_running_matter(line: _LayoutLine) -> bool:
        normalized = re.sub(r"\s+", " ", line.text).strip().casefold()
        near_edge = (
            line.bbox.y0 < line.page_height * 0.055
            or line.bbox.y1 > line.page_height * 0.94
        )
        venue_line = any(
            marker in normalized
            for marker in ("nature | vol", "proceedings of", "arxiv preprint")
        )
        article_header = normalized == "article"
        page_number = bool(re.fullmatch(r"\d{1,4}", normalized))
        return venue_line or article_header or (
            line.page_number > 1 and near_edge and page_number
        )

    @staticmethod
    def _is_page_number_line(line: _LayoutLine) -> bool:
        """Detect a standalone page number (or page/total) printed in the header/footer band."""
        normalized = re.sub(r"\s+", " ", line.text).strip().casefold()
        if not re.fullmatch(
            r"\d{1,4}"
            r"|page\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?"
            r"|\d{1,4}\s*/\s*\d{1,4}",
            normalized,
        ):
            return False
        return (
            line.bbox.y0 < line.page_height * 0.07
            or line.bbox.y1 > line.page_height * 0.93
        )

    def _infer_heading_levels(
        self,
        headings: list[_LayoutLine],
        body_font_size: float,
    ) -> dict[tuple[int, int, str], int]:
        style_sizes = sorted(
            {
                round(line.font_size, 1)
                for line in headings
                if line.font_size >= body_font_size * 1.02
            },
            reverse=True,
        )
        result: dict[tuple[int, int, str], int] = {}
        for line in headings:
            index, _ = self._heading_parts(line.text)
            if index is not None:
                level = self._numbering_level(index)
            elif line.font_size >= body_font_size * 1.02 and style_sizes:
                level = min(style_sizes.index(round(line.font_size, 1)) + 1, 4)
            elif line.bold:
                level = min(len(style_sizes) + 1, 4)
            else:
                level = 1
            result[self._line_key(line)] = max(1, level)
        return result

    def _heading_parts(self, text: str) -> tuple[str | None, str]:
        numbered = self._valid_numbered_heading(text.strip())
        if numbered is None:
            return None, text.strip().rstrip(":")
        return numbered[0], numbered[1]

    @classmethod
    def _merge_text_blocks(cls, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        merged: list[DocumentBlock] = []
        for block in blocks:
            previous = merged[-1] if merged else None
            if not isinstance(block, PageTextBlock) or not isinstance(previous, PageTextBlock):
                merged.append(block)
                continue
            if block.bbox is None or previous.bbox is None:
                merged.append(block)
                continue
            same_page = block.page_number == previous.page_number
            same_column = abs(block.bbox.x0 - previous.bbox.x0) <= 35
            if same_page and same_column:
                font_size = next(
                    (span.font_size for span in block.spans if span.font_size is not None),
                    10.0,
                )
                vertical_gap = block.bbox.y0 - previous.bbox.y1
                if not -1 <= vertical_gap <= font_size * 0.7:
                    merged.append(block)
                    continue
            elif not cls._sentence_continues(previous.text, block.text):
                merged.append(block)
                continue
            if previous.text.endswith("-") and block.text[:1].islower():
                joined = f"{previous.text[:-1]}{block.text}"
            else:
                joined = f"{previous.text} {block.text}"
            merged_bbox = (
                BoundingBox(
                    x0=min(previous.bbox.x0, block.bbox.x0),
                    y0=previous.bbox.y0,
                    x1=max(previous.bbox.x1, block.bbox.x1),
                    y1=block.bbox.y1,
                )
                if same_page and same_column
                else previous.bbox
            )
            merged[-1] = previous.model_copy(
                update={
                    "text": joined,
                    "bbox": merged_bbox,
                    "spans": [*previous.spans, *block.spans],
                    "parse_confidence": min(
                        previous.parse_confidence,
                        block.parse_confidence,
                    ),
                }
            )
        return merged

    @staticmethod
    def _sentence_continues(previous_text: str, next_text: str) -> bool:
        previous = previous_text.rstrip()
        if not previous or not next_text or not next_text[:1].islower():
            return False
        return re.search(r'[.!?。！？]["\')\]]*\s*$', previous) is None

    @staticmethod
    def _numbering_level(index: str) -> int:
        if re.fullmatch(r"[IVXLCDM]+", index):
            return 1
        return index.count(".") + 1

    @staticmethod
    def _line_key(line: _LayoutLine) -> tuple[int, int, str]:
        return (line.page_number, round(line.bbox.y0), line.text)

    def _linked_page_objects(self, page: _LayoutPage) -> list[DocumentBlock]:
        captions = [line for line in page.lines if self._caption_parts(line.text) is not None]
        objects: list[DocumentBlock] = []
        for raw in [*page.tables, *page.figures, *page.code_blocks]:
            if isinstance(raw, CodeBlock):
                objects.append(raw)
                continue
            target = PaperBlockType.TABLE if isinstance(raw, TableBlock) else PaperBlockType.FIGURE
            caption_line = self._nearest_caption(raw.bbox, captions, target)
            caption = caption_line.text if caption_line is not None else None
            caption_parts = self._caption_parts(caption)
            label = caption_parts[1] if caption_parts is not None else None
            update: dict[str, Any] = {
                "block_id": f"page:{page.page_number}:{target.value}:{len(objects) + 1}",
                "caption": caption,
                "object_label": label,
            }
            if isinstance(raw, FigureBlock):
                update["text"] = caption or f"Figure (page {raw.page_number})"
            else:
                already_prefixed = caption is not None and raw.text.strip() == caption
                prefix = "" if already_prefixed else (f"{caption}\n" if caption else "")
                update["text"] = f"{prefix}{raw.text}".strip()
            objects.append(
                cast(
                    DocumentBlock,
                    raw.model_copy(update=update),
                )
            )
        return sorted(objects, key=self._block_y)

    def _nearest_caption(
        self,
        bbox: BoundingBox | None,
        captions: list[_LayoutLine],
        target: PaperBlockType,
    ) -> _LayoutLine | None:
        if bbox is None:
            return None
        matching = [
            line
            for line in captions
            if (parts := self._caption_parts(line.text)) is not None and parts[0] is target
        ]
        if not matching:
            return None
        nearest = min(
            matching,
            key=lambda line: min(
                abs(line.bbox.y0 - bbox.y1),
                abs(bbox.y0 - line.bbox.y1),
            ),
        )
        distance = min(abs(nearest.bbox.y0 - bbox.y1), abs(bbox.y0 - nearest.bbox.y1))
        return nearest if distance <= nearest.page_height * 0.15 else None

    def _line_block(self, paper_id: str, line: _LayoutLine, index: int) -> DocumentBlock:
        block_id = f"{paper_id}:block:{line.page_number:04d}:{index + 1:04d}"
        caption = self._caption_parts(line.text)
        if caption is not None:
            return CaptionBlock(
                block_id=block_id,
                page_number=line.page_number,
                bbox=line.bbox,
                reading_order=index,
                parse_confidence=0.95,
                text=line.text,
                object_label=caption[1],
                target_type=caption[0],
            )
        if self._looks_like_equation(line.text):
            label_match = re.search(r"\(([A-Z]?\d+(?:\.\d+)?)\)\s*$", line.text)
            return EquationBlock(
                block_id=block_id,
                page_number=line.page_number,
                bbox=line.bbox,
                reading_order=index,
                parse_confidence=0.65,
                text=line.text,
                raw_text=line.text,
                object_label=label_match.group(1) if label_match else None,
            )
        return PageTextBlock(
            block_id=block_id,
            page_number=line.page_number,
            bbox=line.bbox,
            reading_order=index,
            parse_confidence=0.98,
            text=line.text,
            spans=list(line.spans),
        )

    @staticmethod
    def _caption_parts(text: str | None) -> tuple[PaperBlockType, str] | None:
        if not text:
            return None
        match = re.match(
            r"^\s*(?:extended\s+data\s+)?"
            r"(?P<kind>table|figure|fig\.?)\s*"
            r"(?P<label>[A-Z]?\d+(?:\.\d+)?)\b",
            text,
            re.I,
        )
        if match is None:
            return None
        kind = match.group("kind").casefold()
        block_type = PaperBlockType.TABLE if kind == "table" else PaperBlockType.FIGURE
        return block_type, match.group("label")

    @staticmethod
    def _looks_like_equation(text: str) -> bool:
        stripped = text.strip()
        if len(stripped) > 300:
            return False
        math_symbols = len(re.findall(r"[=≤≥∑∏∫√∞λθαβγμσ±×÷∈∉∪∩→←]", stripped))
        has_number = re.search(r"\([A-Z]?\d+(?:\.\d+)?\)\s*$", stripped) is not None
        return math_symbols >= 2 or (math_symbols >= 1 and has_number)

    @staticmethod
    def _inside_objects(line: _LayoutLine, objects: list[DocumentBlock]) -> bool:
        center_x = (line.bbox.x0 + line.bbox.x1) / 2
        center_y = (line.bbox.y0 + line.bbox.y1) / 2
        return any(
            block.bbox is not None
            and block.bbox.x0 <= center_x <= block.bbox.x1
            and block.bbox.y0 <= center_y <= block.bbox.y1
            for block in objects
            if block.block_type in {PaperBlockType.TABLE, PaperBlockType.FIGURE}
        )

    @staticmethod
    def _block_y(block: DocumentBlock) -> float:
        return block.bbox.y0 if block.bbox is not None else float("inf")

    @staticmethod
    def _semantic_role(title: str) -> str | None:
        normalized = title.casefold()
        roles = {
            "abstract": ("abstract",),
            "introduction": ("introduction", "overview"),
            "related_work": ("related work", "background", "prior work"),
            "methods": ("method", "approach", "architecture", "implementation"),
            "experiments": ("experiment", "evaluation", "benchmark", "setup"),
            "results": ("result", "analysis", "finding"),
            "limitations": ("limitation", "threats to validity"),
            "discussion": ("discussion",),
            "conclusion": ("conclusion", "future work"),
            "references": ("references", "bibliography"),
            "appendix": ("appendix", "supplement"),
        }
        return next(
            (role for role, hints in roles.items() if any(hint in normalized for hint in hints)),
            None,
        )

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
            or title.endswith((".", "!"))
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
