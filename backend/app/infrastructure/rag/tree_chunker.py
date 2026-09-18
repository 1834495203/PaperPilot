import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from app.domain.rag import (
    BoundingBox,
    CaptionBlock,
    DocumentBlock,
    EvidenceSpan,
    FigureBlock,
    PageTextBlock,
    PaperBlockType,
    PaperSection,
    ParsedPaperDocument,
    TableBlock,
    TreeIndexNode,
    TreeNodeType,
)


@dataclass(slots=True)
class _MutableNode:
    node_id: str
    paper_id: str
    node_type: TreeNodeType
    title: str
    parent_id: str | None
    level: int
    section_path: list[str]
    semantic_role: str | None
    block_types: list[PaperBlockType]
    object_labels: list[str]
    text: str
    embedding_text: str
    page_start: int | None
    page_end: int | None
    figure_asset: str | None = None
    figure_caption: str | None = None
    raw_asset_ref: str | None = None
    table_rows: list[list[str]] | None = None
    spans: list[EvidenceSpan] = field(default_factory=list)
    children_ids: list[str] = field(default_factory=list)

    def freeze(self) -> TreeIndexNode:
        return TreeIndexNode(
            node_id=self.node_id,
            paper_id=self.paper_id,
            node_type=self.node_type,
            title=self.title,
            parent_id=self.parent_id,
            children_ids=self.children_ids,
            level=self.level,
            section_path=self.section_path,
            semantic_role=self.semantic_role,
            block_types=self.block_types,
            object_labels=self.object_labels,
            text=self.text,
            embedding_text=self.embedding_text,
            page_start=self.page_start,
            page_end=self.page_end,
            figure_asset=self.figure_asset,
            figure_caption=self.figure_caption,
            raw_asset_ref=self.raw_asset_ref,
            table_rows=self.table_rows,
            spans=self.spans,
        )


@dataclass(frozen=True, slots=True)
class _TextUnit:
    page_number: int
    text: str
    block_type: PaperBlockType
    object_label: str | None = None
    target_type: PaperBlockType | None = None
    figure_asset: str | None = None
    figure_caption: str | None = None
    raw_asset_ref: str | None = None
    table_rows: list[list[str]] | None = None
    bbox: BoundingBox | None = None


class TreeRagChunker:
    """Create TreeRAG-style nodes with ancestor-title-prefixed embedding text."""

    VERSION = "tree-rag-4"

    def __init__(
        self,
        *,
        max_chunk_chars: int = 1_800,
        overlap_sentences: int = 1,
    ) -> None:
        if max_chunk_chars < 200:
            raise ValueError("max_chunk_chars must be at least 200")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be non-negative")
        self._max_chunk_chars = max_chunk_chars
        self._overlap_sentences = overlap_sentences

    def chunk(self, document: ParsedPaperDocument) -> list[TreeIndexNode]:
        root_id = f"{document.paper_id}:root"
        root_embedding_text = self._root_embedding_text(document)
        mutable_nodes: dict[str, _MutableNode] = {
            root_id: _MutableNode(
                node_id=root_id,
                paper_id=document.paper_id,
                node_type=TreeNodeType.ROOT,
                title=document.title,
                parent_id=None,
                level=0,
                section_path=[],
                semantic_role=None,
                block_types=[],
                object_labels=[],
                text=document.title,
                embedding_text=root_embedding_text,
                page_start=1,
                page_end=document.page_count,
            )
        }
        section_by_id = {section.section_id: section for section in document.sections}
        path_cache: dict[str, list[str]] = {}

        def section_path(section_id: str) -> list[str]:
            cached = path_cache.get(section_id)
            if cached is not None:
                return cached
            section = section_by_id[section_id]
            parent_path = (
                section_path(section.parent_section_id)
                if section.parent_section_id in section_by_id
                else []
            )
            label = f"{section.index} {section.title}" if section.index else section.title
            resolved = [*parent_path, label]
            path_cache[section_id] = resolved
            return resolved

        for section in document.sections:
            path = section_path(section.section_id)
            parent_id = section.parent_section_id or root_id
            embedding_text = self._section_embedding_text(document.title, path, section)
            mutable_nodes[section.section_id] = _MutableNode(
                node_id=section.section_id,
                paper_id=document.paper_id,
                node_type=TreeNodeType.SECTION,
                title=path[-1],
                parent_id=parent_id,
                level=section.level,
                section_path=path,
                semantic_role=section.semantic_role,
                block_types=[],
                object_labels=[],
                text=section.title,
                embedding_text=embedding_text,
                page_start=section.page_start,
                page_end=section.page_end,
            )

        for section in document.sections:
            for index, units in enumerate(self._section_chunks(section.blocks), start=1):
                node_id = f"{section.section_id}:chunk:{index:04d}"
                text = self._join_units(units)
                pages = [unit.page_number for unit in units]
                block_types = list(dict.fromkeys(unit.block_type for unit in units))
                object_labels = list(
                    dict.fromkeys(
                        unit.object_label for unit in units if unit.object_label is not None
                    )
                )
                figure_asset = next(
                    (unit.figure_asset for unit in units if unit.figure_asset is not None),
                    None,
                )
                figure_caption = next(
                    (unit.figure_caption for unit in units if unit.figure_caption is not None),
                    None,
                )
                raw_asset_ref = next(
                    (unit.raw_asset_ref for unit in units if unit.raw_asset_ref is not None),
                    None,
                )
                table_rows = next(
                    (unit.table_rows for unit in units if unit.table_rows is not None),
                    None,
                )
                spans = [
                    span
                    for unit in units
                    if (span := self._evidence_span(unit, document.page_dimensions)) is not None
                ]
                path = section_path(section.section_id)
                mutable_nodes[node_id] = _MutableNode(
                    node_id=node_id,
                    paper_id=document.paper_id,
                    node_type=TreeNodeType.CHUNK,
                    title=path[-1],
                    parent_id=section.section_id,
                    level=section.level + 1,
                    section_path=path,
                    semantic_role=section.semantic_role,
                    block_types=block_types,
                    object_labels=object_labels,
                    text=text,
                    embedding_text=self._prefixed_text(document.title, path, text),
                    page_start=min(pages),
                    page_end=max(pages),
                    figure_asset=figure_asset,
                    figure_caption=figure_caption,
                    raw_asset_ref=raw_asset_ref,
                    table_rows=table_rows,
                    spans=spans,
                )

        for node in mutable_nodes.values():
            if node.parent_id is not None and node.parent_id in mutable_nodes:
                mutable_nodes[node.parent_id].children_ids.append(node.node_id)
        return [node.freeze() for node in mutable_nodes.values()]

    _ABBREVIATIONS = (
        "et al.",
        "e.g.",
        "i.e.",
        "etc.",
        "cf.",
        "vs.",
        "approx.",
        "fig.",
        "figs.",
        "tab.",
        "tabs.",
        "eq.",
        "eqs.",
        "eqn.",
        "eqns.",
        "sec.",
        "secs.",
        "ref.",
        "refs.",
        "no.",
        "nos.",
        "vol.",
        "vols.",
        "pp.",
        "dr.",
        "mr.",
        "ms.",
        "prof.",
        "st.",
        "inc.",
        "ltd.",
        "co.",
        "corp.",
    )

    def _section_chunks(self, blocks: Sequence[DocumentBlock]) -> list[list[_TextUnit]]:
        units = self._build_units(blocks)
        packed = self._pack_text(units)
        if self._overlap_sentences > 0:
            packed = self._overlap_chunks(packed)
        return packed

    def _build_units(self, blocks: Sequence[DocumentBlock]) -> list[_TextUnit]:
        units: list[_TextUnit] = []
        object_index: dict[tuple[PaperBlockType, str], int] = {}
        caption_indices: list[int] = []
        for block in blocks:
            if isinstance(block, CaptionBlock):
                units.append(
                    _TextUnit(
                        page_number=block.page_number,
                        text=block.text,
                        block_type=PaperBlockType.CAPTION,
                        object_label=block.object_label,
                        target_type=block.target_type,
                        bbox=block.bbox,
                    )
                )
                caption_indices.append(len(units) - 1)
                continue
            if isinstance(block, PageTextBlock):
                units.extend(self._text_units(block))
                continue
            index = len(units)
            units.append(self._object_unit(block))
            if isinstance(block, (FigureBlock, TableBlock)) and block.object_label is not None:
                object_index[(block.block_type, block.object_label)] = index

        drop: set[int] = set()
        for caption_index in caption_indices:
            caption_unit = units[caption_index]
            target = (
                object_index.get((caption_unit.target_type, caption_unit.object_label))
                if caption_unit.object_label is not None
                and caption_unit.target_type is not None
                else None
            )
            if target is None:
                continue
            units[target] = self._attach_caption(units[target], caption_unit)
            drop.add(caption_index)
        return [unit for index, unit in enumerate(units) if index not in drop]

    def _attach_caption(self, unit: _TextUnit, caption: _TextUnit) -> _TextUnit:
        caption_text = caption.text.strip()
        if not caption_text or caption_text in unit.text:
            return unit
        if unit.block_type is PaperBlockType.FIGURE:
            figure_caption = unit.figure_caption or caption_text
            text = (
                caption_text
                if unit.text == figure_caption
                else f"{caption_text}\n{unit.text}"
            )
            return replace(unit, text=text, figure_caption=figure_caption)
        if unit.block_type is PaperBlockType.TABLE:
            return replace(unit, text=f"{caption_text}\n{unit.text}")
        return unit

    def _object_unit(self, block: DocumentBlock) -> _TextUnit:
        is_figure = block.block_type is PaperBlockType.FIGURE
        is_table = block.block_type is PaperBlockType.TABLE
        return _TextUnit(
            page_number=block.page_number,
            text=block.text,
            block_type=block.block_type,
            object_label=getattr(block, "object_label", None),
            figure_asset=getattr(block, "asset_ref", None) if is_figure else None,
            figure_caption=getattr(block, "caption", None) if is_figure else None,
            raw_asset_ref=getattr(block, "raw_asset_ref", None) if is_figure else None,
            table_rows=getattr(block, "rows", None) if is_table else None,
            bbox=block.bbox,
        )

    def _text_units(self, block: DocumentBlock) -> list[_TextUnit]:
        units: list[_TextUnit] = []
        for sentence in self._split_sentences(block.text):
            for piece in self._hard_split(sentence):
                units.append(
                    _TextUnit(
                        page_number=block.page_number,
                        text=piece,
                        block_type=PaperBlockType.TEXT,
                        bbox=block.bbox,
                    )
                )
        return units

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        protected = text
        for abbreviation in cls._ABBREVIATIONS:
            protected = re.sub(
                re.escape(abbreviation),
                abbreviation.replace(".", "\u0000"),
                protected,
                flags=re.IGNORECASE,
            )
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?。！？])\s+", protected)
            if item.strip()
        ]
        return [item.replace("\u0000", ".") for item in sentences]

    def _hard_split(self, sentence: str) -> list[str]:
        if len(sentence) <= self._max_chunk_chars:
            return [sentence]
        return [
            sentence[start : start + self._max_chunk_chars]
            for start in range(0, len(sentence), self._max_chunk_chars)
        ]

    def _pack_text(self, units: list[_TextUnit]) -> list[list[_TextUnit]]:
        packed: list[list[_TextUnit]] = []
        current: list[_TextUnit] = []
        current_length = 0
        for unit in units:
            if unit.block_type is not PaperBlockType.TEXT:
                if current:
                    packed.append(current)
                    current = []
                    current_length = 0
                packed.append([unit])
                continue
            separator_length = 1 if current else 0
            if (
                current
                and current_length + separator_length + len(unit.text) > self._max_chunk_chars
            ):
                packed.append(current)
                current = []
                current_length = 0
                separator_length = 0
            current.append(unit)
            current_length += separator_length + len(unit.text)
        if current:
            packed.append(current)
        return packed

    def _overlap_chunks(self, chunks: list[list[_TextUnit]]) -> list[list[_TextUnit]]:
        result: list[list[_TextUnit]] = []
        previous_text: list[_TextUnit] | None = None
        for chunk in chunks:
            is_text = bool(chunk) and all(
                unit.block_type is PaperBlockType.TEXT for unit in chunk
            )
            overlapped = chunk
            if is_text and previous_text is not None:
                overlapped = [*previous_text[-self._overlap_sentences :], *chunk]
            result.append(overlapped)
            previous_text = chunk if is_text else None
        return result

    @staticmethod
    def _join_units(units: list[_TextUnit]) -> str:
        if units and all(unit.block_type is PaperBlockType.TEXT for unit in units):
            return " ".join(unit.text for unit in units)
        return "\n\n".join(unit.text for unit in units)

    @staticmethod
    def _prefixed_text(paper_title: str, section_path: list[str], text: str) -> str:
        # TreeRAG Equation (1): ancestor title content is concatenated before the node text.
        return "\n".join([paper_title, *section_path, text]).strip()

    @staticmethod
    def _evidence_span(
        unit: _TextUnit,
        page_dimensions: dict[int, tuple[float, float]],
    ) -> EvidenceSpan | None:
        bbox = unit.bbox
        if bbox is None:
            return None
        dimensions = page_dimensions.get(unit.page_number)
        if dimensions is not None and dimensions[0] > 0 and dimensions[1] > 0:
            bbox = BoundingBox(
                x0=bbox.x0 / dimensions[0],
                y0=bbox.y0 / dimensions[1],
                x1=bbox.x1 / dimensions[0],
                y1=bbox.y1 / dimensions[1],
            )
        return EvidenceSpan(text=unit.text, page_number=unit.page_number, bbox=bbox)

    @staticmethod
    def _root_embedding_text(document: ParsedPaperDocument) -> str:
        parts = [document.title]
        if document.metadata.abstract:
            parts.append(f"Abstract: {document.metadata.abstract}")
        if document.metadata.keywords:
            parts.append(f"Keywords: {', '.join(document.metadata.keywords)}")
        outline = [
            f"{section.index} {section.title}" if section.index else section.title
            for section in document.sections
        ]
        if outline:
            parts.append("Sections: " + "; ".join(outline))
        return "\n".join(parts)

    @classmethod
    def _section_embedding_text(
        cls,
        paper_title: str,
        section_path: list[str],
        section: PaperSection,
    ) -> str:
        # The section vector is a routing summary, while the original section title
        # remains unchanged for display and citation.
        representative = "\n".join(
            block.text.strip() for block in section.blocks
        ).strip()
        representative = representative[:1_500]
        role_text = (
            f"Semantic role: {section.semantic_role}" if section.semantic_role else ""
        )
        return cls._prefixed_text(
            paper_title,
            section_path,
            "\n".join(item for item in (role_text, representative) if item),
        )
