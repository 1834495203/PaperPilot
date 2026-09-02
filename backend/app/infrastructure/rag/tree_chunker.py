import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.domain.rag import PageTextBlock, ParsedPaperDocument, TreeIndexNode, TreeNodeType


@dataclass(slots=True)
class _MutableNode:
    node_id: str
    paper_id: str
    node_type: TreeNodeType
    title: str
    parent_id: str | None
    level: int
    section_path: list[str]
    text: str
    embedding_text: str
    page_start: int | None
    page_end: int | None
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
            text=self.text,
            embedding_text=self.embedding_text,
            page_start=self.page_start,
            page_end=self.page_end,
        )


@dataclass(frozen=True, slots=True)
class _TextUnit:
    page_number: int
    text: str


class TreeRagChunker:
    """Create TreeRAG-style nodes with ancestor-title-prefixed embedding text."""

    def __init__(self, *, max_chunk_chars: int = 1_800) -> None:
        if max_chunk_chars < 200:
            raise ValueError("max_chunk_chars must be at least 200")
        self._max_chunk_chars = max_chunk_chars

    def chunk(self, document: ParsedPaperDocument) -> list[TreeIndexNode]:
        root_id = f"{document.paper_id}:root"
        mutable_nodes: dict[str, _MutableNode] = {
            root_id: _MutableNode(
                node_id=root_id,
                paper_id=document.paper_id,
                node_type=TreeNodeType.ROOT,
                title=document.title,
                parent_id=None,
                level=0,
                section_path=[],
                text=document.title,
                embedding_text=document.title,
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
            embedding_text = self._prefixed_text(document.title, path, "")
            mutable_nodes[section.section_id] = _MutableNode(
                node_id=section.section_id,
                paper_id=document.paper_id,
                node_type=TreeNodeType.SECTION,
                title=path[-1],
                parent_id=parent_id,
                level=section.level,
                section_path=path,
                text=section.title,
                embedding_text=embedding_text,
                page_start=section.page_start,
                page_end=section.page_end,
            )

        for section in document.sections:
            for index, units in enumerate(self._section_chunks(section.blocks), start=1):
                node_id = f"{section.section_id}:chunk:{index:04d}"
                text = "\n\n".join(unit.text for unit in units)
                pages = [unit.page_number for unit in units]
                path = section_path(section.section_id)
                mutable_nodes[node_id] = _MutableNode(
                    node_id=node_id,
                    paper_id=document.paper_id,
                    node_type=TreeNodeType.CHUNK,
                    title=path[-1],
                    parent_id=section.section_id,
                    level=section.level + 1,
                    section_path=path,
                    text=text,
                    embedding_text=self._prefixed_text(document.title, path, text),
                    page_start=min(pages),
                    page_end=max(pages),
                )

        for node in mutable_nodes.values():
            if node.parent_id is not None and node.parent_id in mutable_nodes:
                mutable_nodes[node.parent_id].children_ids.append(node.node_id)
        return [node.freeze() for node in mutable_nodes.values()]

    def _section_chunks(self, blocks: Sequence[PageTextBlock]) -> list[list[_TextUnit]]:
        units: list[_TextUnit] = []
        for block in blocks:
            page_number = block.page_number
            block_text = block.text
            paragraphs = [item.strip() for item in block_text.split("\n\n") if item.strip()]
            for paragraph in paragraphs:
                units.extend(
                    _TextUnit(page_number=page_number, text=part)
                    for part in self._split_long_text(paragraph)
                )
        packed: list[list[_TextUnit]] = []
        current: list[_TextUnit] = []
        current_length = 0
        for unit in units:
            separator_length = 2 if current else 0
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

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self._max_chunk_chars:
            return [text]
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?。！？])\s+", text)
            if item.strip()
        ]
        if len(sentences) == 1:
            return [
                text[start : start + self._max_chunk_chars]
                for start in range(0, len(text), self._max_chunk_chars)
            ]
        parts: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > self._max_chunk_chars:
                if current:
                    parts.append(current)
                    current = ""
                parts.extend(
                    sentence[start : start + self._max_chunk_chars]
                    for start in range(0, len(sentence), self._max_chunk_chars)
                )
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > self._max_chunk_chars:
                parts.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts

    @staticmethod
    def _prefixed_text(paper_title: str, section_path: list[str], text: str) -> str:
        # TreeRAG Equation (1): ancestor title content is concatenated before the node text.
        return "\n".join([paper_title, *section_path, text]).strip()
