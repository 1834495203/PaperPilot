import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, ClassVar, Literal

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


class RetrievalStrategy(StrEnum):
    """Task-level retrieval strategy; each value owns its own candidate budget."""

    SINGLE_PAPER = "single_paper"
    MULTI_PAPER = "multi_paper"
    CORPUS_SURVEY = "corpus_survey"


class RetrievalSource(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    TREE_EXPANSION = "tree_expansion"


class CoverageStatus(StrEnum):
    """Outcome of one paper x dimension coverage cell.

    ``CANDIDATE`` means retrieval delivered a chunk for the cell but nothing has
    verified that it answers the dimension, ``MISSING`` means no evidence was
    obtained and one may exist, and ``NOT_STATED`` is an evidence-backed verdict
    that the paper does not address the dimension at all. Only an evidence reader
    may set ``COVERED`` or ``NOT_STATED``.
    """

    CANDIDATE = "candidate"
    COVERED = "covered"
    MISSING = "missing"
    NOT_STATED = "not_stated"


class PaperBlockType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    CAPTION = "caption"
    CODE = "code"


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


class EvidenceSpan(BaseModel):
    """A page-anchored text span used to link evidence back to its source region."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    bbox: BoundingBox


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
    asset_ref: str | None = None
    raw_asset_ref: str | None = None


class EquationBlock(DocumentBlockBase):
    block_type: Literal[PaperBlockType.EQUATION] = PaperBlockType.EQUATION
    object_label: str | None = None
    raw_text: str
    latex: str | None = None


class CaptionBlock(DocumentBlockBase):
    block_type: Literal[PaperBlockType.CAPTION] = PaperBlockType.CAPTION
    object_label: str | None = None
    target_type: PaperBlockType | None = None


class CodeBlock(DocumentBlockBase):
    block_type: Literal[PaperBlockType.CODE] = PaperBlockType.CODE
    language: str | None = None


DocumentBlock = Annotated[
    PageTextBlock | TableBlock | FigureBlock | EquationBlock | CaptionBlock | CodeBlock,
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
    page_dimensions: dict[int, tuple[float, float]] = Field(default_factory=dict)

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
    figure_asset: str | None = None
    figure_caption: str | None = None
    raw_asset_ref: str | None = None
    table_rows: list[list[str]] | None = None
    spans: list[EvidenceSpan] = Field(default_factory=list)

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
    index_signature: str | None = None
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


class PaperIndexStatus(BaseModel):
    """Whether one indexed paper still matches the current index fingerprint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    title: str
    chunk_count: int = Field(ge=0)
    index_signature: str | None = None
    current_signature: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_rebuild(self) -> bool:
        """True only when a recorded fingerprint disagrees with the current one."""

        if self.current_signature is None or self.index_signature is None:
            return False
        return self.index_signature != self.current_signature

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint_recorded(self) -> bool:
        return self.index_signature is not None


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
    figure_asset: str | None = None
    figure_caption: str | None = None
    raw_asset_ref: str | None = None
    table_rows: list[list[str]] | None = None
    spans: list[EvidenceSpan] = Field(default_factory=list)


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


class TreeKeywordMatch(BaseModel):
    """A hit produced by the lexical channel, independent of vector recall."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node: TreeIndexNode
    keyword_score: float = Field(ge=0, le=1)
    matched_terms: list[str] = Field(default_factory=list)


class KeywordSearchResult(BaseModel):
    """Lexical recall output plus its own bounded-pool diagnostics.

    ``truncated_terms`` lists query terms whose substring filter returned more
    records than the store fetched, so the pool - and therefore the BM25 term
    statistics - was incomplete for them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: list[TreeKeywordMatch] = Field(default_factory=list)
    pool_size: int = Field(default=0, ge=0)
    truncated_terms: list[str] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    """One retrieval sub-question with its own scope, mode and coverage dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=2_000)
    mode: RetrievalMode
    paper_ids: list[str] = Field(default_factory=list)
    dimension: str | None = Field(default=None, max_length=200)


class RetrievalBudget(BaseModel):
    """Candidate and evidence budget resolved from the task strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_papers: int = Field(ge=1)
    sections_per_paper: int = Field(ge=1)
    initial_top_k: int = Field(ge=1)
    global_fallback_top_k: int = Field(ge=1)
    keyword_top_k: int = Field(ge=1)
    max_expanded_per_hit: int = Field(ge=1)
    rerank_candidate_limit: int = Field(ge=1)
    max_chunks_per_paper: int = Field(ge=1)
    final_top_k: int = Field(ge=1)


class CoverageCell(BaseModel):
    """One paper x dimension cell of the multi-paper coverage check.

    ``evidence_ids`` are candidate chunk node IDs delivered by retrieval, while
    ``verified_evidence_ids`` are Evidence Library IDs that an evidence reader
    cited to confirm the cell is answered.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    dimension: str
    status: CoverageStatus
    evidence_ids: list[str] = Field(default_factory=list)
    verified_evidence_ids: list[str] = Field(default_factory=list)
    note: str | None = None


class CoverageMatrix(BaseModel):
    """Paper x dimension coverage of the current evidence set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    DEFAULT_DIMENSION: ClassVar[str] = "content"

    dimensions: list[str] = Field(default_factory=list)
    cells: list[CoverageCell] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def paper_ids(self) -> list[str]:
        return list(dict.fromkeys(cell.paper_id for cell in self.cells))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def candidate_cell_count(self) -> int:
        """Cells with a retrieved candidate that no evidence reader has verified."""

        return sum(cell.status is CoverageStatus.CANDIDATE for cell in self.cells)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def covered_cell_count(self) -> int:
        return sum(cell.status is CoverageStatus.COVERED for cell in self.cells)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unresolved_cell_count(self) -> int:
        return sum(cell.status is not CoverageStatus.COVERED for cell in self.cells)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coverage_ratio(self) -> float:
        """Verified coverage only; candidate cells do not count as answered."""

        if not self.cells:
            return 0.0
        return self.covered_cell_count / len(self.cells)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def candidate_ratio(self) -> float:
        """Cells where retrieval at least produced something to judge."""

        if not self.cells:
            return 0.0
        return (self.covered_cell_count + self.candidate_cell_count) / len(self.cells)

    def cell_for(self, paper_id: str, dimension: str) -> CoverageCell | None:
        return next(
            (
                cell
                for cell in self.cells
                if cell.paper_id == paper_id and cell.dimension == dimension
            ),
            None,
        )

    def unresolved_cells(self) -> list[CoverageCell]:
        return [cell for cell in self.cells if cell.status is not CoverageStatus.COVERED]

    def candidate_cells(self) -> list[CoverageCell]:
        return [cell for cell in self.cells if cell.status is CoverageStatus.CANDIDATE]

    def with_cell(self, updated: CoverageCell) -> "CoverageMatrix":
        cells = [
            updated
            if cell.paper_id == updated.paper_id and cell.dimension == updated.dimension
            else cell
            for cell in self.cells
        ]
        if not any(
            cell.paper_id == updated.paper_id and cell.dimension == updated.dimension
            for cell in self.cells
        ):
            cells.append(updated)
        dimensions = list(self.dimensions)
        if updated.dimension not in dimensions:
            dimensions.append(updated.dimension)
        return self.model_copy(update={"cells": cells, "dimensions": dimensions})

    @classmethod
    def build(
        cls,
        *,
        paper_ids: list[str],
        dimensions: list[str],
    ) -> "CoverageMatrix":
        effective_dimensions = dimensions or [cls.DEFAULT_DIMENSION]
        return cls(
            dimensions=effective_dimensions,
            cells=[
                CoverageCell(
                    paper_id=paper_id,
                    dimension=dimension,
                    status=CoverageStatus.MISSING,
                )
                for paper_id in paper_ids
                for dimension in effective_dimensions
            ],
        )


class IndexSignature(BaseModel):
    """Identifies the models and code versions that produced a vector index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    embedding_model: str = Field(min_length=1)
    embedding_dimensions: int | None = Field(default=None, ge=1)
    chunker_version: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    schema_version: int = Field(default=4, ge=1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fingerprint(self) -> str:
        payload = "|".join(
            [
                self.embedding_model,
                str(self.embedding_dimensions or 0),
                self.chunker_version,
                self.parser_version,
                str(self.schema_version),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    node_id: str
    paper_id: str
    paper_title: str | None = None
    section_path: list[str]
    semantic_role: str | None = None
    block_types: list[PaperBlockType] = Field(default_factory=list)
    object_labels: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    text: str
    vector_score: float
    ranking_score: float
    rerank_score: float | None = None
    rerank_query: str | None = None
    keyword_score: float | None = None
    fused_score: float | None = None
    matched_terms: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)
    matched_dimensions: list[str] = Field(default_factory=list)
    source: RetrievalSource
    expanded_from: str | None = None
    figure_asset: str | None = None
    figure_caption: str | None = None
    raw_asset_ref: str | None = None
    table_rows: list[list[str]] | None = None
    spans: list[EvidenceSpan] = Field(default_factory=list)


class TreeRetrievalReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    mode: RetrievalMode
    paper_ids: list[str]
    strategy: RetrievalStrategy = RetrievalStrategy.SINGLE_PAPER
    budget: RetrievalBudget | None = None
    queries: list[RetrievalQuery] = Field(default_factory=list)
    searched_globally: bool = False
    candidate_paper_ids: list[str] = Field(default_factory=list)
    missing_paper_ids: list[str] = Field(default_factory=list)
    coverage: CoverageMatrix | None = None
    initial_hit_count: int = Field(ge=0)
    expanded_candidate_count: int = Field(ge=0)
    deduplicated_candidate_count: int = Field(default=0, ge=0)
    mmr_candidate_count: int = Field(default=0, ge=0)
    keyword_candidate_count: int = Field(default=0, ge=0)
    keyword_pool_size: int = Field(default=0, ge=0)
    keyword_truncated_terms: list[str] = Field(default_factory=list)
    fused_candidate_count: int = Field(default=0, ge=0)
    reranker_name: str | None = None
    reranker_applied: bool = False
    reranker_error: str | None = None
    hits: list[RetrievalHit]


class Evidence(BaseModel):
    """One immutable, fully traceable piece of retrieved source evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^E-[0-9a-f]{12}$")
    paper_id: str
    paper_title: str
    chunk_id: str
    section_path: list[str]
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    raw_text: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    retrieval_score: float
    rerank_score: float | None = None
    evidence_score: float | None = Field(default=None, ge=0, le=10)
    supported_claims: list[str] = Field(default_factory=list)
    spans: list[EvidenceSpan] = Field(default_factory=list)


class EvidenceLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: str = Field(min_length=1)
    evidence: list[Evidence]
