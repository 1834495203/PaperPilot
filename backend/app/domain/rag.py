from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class TreeNodeType(StrEnum):
    ROOT = "root"
    SECTION = "section"
    CHUNK = "chunk"


class RetrievalMode(StrEnum):
    FACT = "fact"
    SUMMARY = "summary"
    METHOD = "method"
    COMPARE = "compare"
    SYNTHESIS = "synthesis"

    @property
    def expands_tree(self) -> bool:
        return self is not RetrievalMode.FACT


class RetrievalSource(StrEnum):
    VECTOR = "vector"
    TREE_EXPANSION = "tree_expansion"


class PaperBlockType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CAPTION = "caption"


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_extents(self) -> "BoundingBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding box extents must be ordered")
        return self


class TextSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    bbox: BoundingBox
    font_name: str | None = None
    font_size: float | None = Field(default=None, gt=0)
    bold: bool = False
    italic: bool = False


class DocumentBlockBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str = ""
    page_number: int = Field(ge=1)
    bbox: BoundingBox | None = None
    reading_order: int = Field(default=0, ge=0)
    parse_confidence: float = Field(default=1.0, ge=0, le=1)
    text: str = Field(min_length=1)


class PageTextBlock(DocumentBlockBase):
    block_type: Literal[PaperBlockType.TEXT] = PaperBlockType.TEXT
    spans: list[TextSpan] = Field(default_factory=list)


class TableBlock(DocumentBlockBase):
    block_type: Literal[PaperBlockType.TABLE] = PaperBlockType.TABLE
    object_label: str | None = None
    caption: str | None = None
    rows: list[list[str]] = Field(default_factory=list)
    markdown: str | None = None


class FigureBlock(DocumentBlockBase):
    block_type: Literal[PaperBlockType.FIGURE] = PaperBlockType.FIGURE
    object_label: str | None = None
    caption: str | None = None


class EquationBlock(DocumentBlockBase):
    block_type: Literal[PaperBlockType.EQUATION] = PaperBlockType.EQUATION
    object_label: str | None = None
    raw_text: str
    latex: str | None = None


class CaptionBlock(DocumentBlockBase):
    block_type: Literal[PaperBlockType.CAPTION] = PaperBlockType.CAPTION
    object_label: str | None = None
    target_type: PaperBlockType | None = None


DocumentBlock = Annotated[
    PageTextBlock | TableBlock | FigureBlock | EquationBlock | CaptionBlock,
    Field(discriminator="block_type"),
]


class PaperSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str
    index: str | None = None
    title: str
    semantic_role: str | None = None
    level: int = Field(ge=1)
    parent_section_id: str | None = None
    blocks: list[DocumentBlock]
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class PaperMetadata(BaseModel):
    """Bibliographic metadata extracted once during ingestion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None


class ParsedPaperDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    metadata: PaperMetadata
    source_path: Path
    page_count: int = Field(ge=1)
    sections: list[PaperSection]

    @property
    def title(self) -> str:
        return self.metadata.title


class TreeIndexNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    paper_id: str
    node_type: TreeNodeType
    title: str
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    level: int = Field(ge=0)
    section_path: list[str]
    semantic_role: str | None = None
    block_types: list[PaperBlockType] = Field(default_factory=list)
    object_labels: list[str] = Field(default_factory=list)
    text: str
    embedding_text: str = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_leaf(self) -> bool:
        return not self.children_ids


class PaperIngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    metadata: PaperMetadata
    page_count: int
    section_count: int
    node_count: int
    chunk_count: int
    vector_collection: str

    @property
    def title(self) -> str:
        return self.metadata.title


class IndexedPaper(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    metadata: PaperMetadata
    original_filename: str
    content_sha256: str
    page_count: int = Field(ge=1)
    section_count: int = Field(ge=0)
    node_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_manifest(cls, value: object) -> object:
        if not isinstance(value, dict) or "metadata" in value:
            return value
        migrated = dict(value)
        title = migrated.pop("title", None)
        if title:
            migrated["metadata"] = {"title": title}
        return migrated

    @property
    def title(self) -> str:
        return self.metadata.title


class PaperTreeNodeView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    node_type: TreeNodeType
    title: str
    parent_id: str | None
    children_ids: list[str]
    level: int = Field(ge=0)
    section_path: list[str]
    semantic_role: str | None = None
    block_types: list[PaperBlockType] = Field(default_factory=list)
    object_labels: list[str] = Field(default_factory=list)
    page_start: int | None
    page_end: int | None
    text_preview: str
    text: str


class IndexedPaperDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper: IndexedPaper
    nodes: list[PaperTreeNodeView]


class IndexedTreeNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node: TreeIndexNode
    embedding: list[float]


class TreeVectorMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node: TreeIndexNode
    vector_score: float


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    node_id: str
    paper_id: str
    section_path: list[str]
    semantic_role: str | None = None
    block_types: list[PaperBlockType] = Field(default_factory=list)
    object_labels: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    text: str
    vector_score: float
    ranking_score: float
    source: RetrievalSource
    expanded_from: str | None = None


class TreeRetrievalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    mode: RetrievalMode
    paper_ids: list[str]
    searched_globally: bool = False
    candidate_paper_ids: list[str] = Field(default_factory=list)
    initial_hit_count: int = Field(ge=0)
    expanded_candidate_count: int = Field(ge=0)
    hits: list[RetrievalHit]
