from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class PaperSource(StrEnum):
    OPENALEX = "openalex"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    ARXIV = "arxiv"


class PaperSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=2, max_length=300, description="A focused academic search query")
    max_results: int = Field(default=5, ge=1, le=10)
    sort_by: str = Field(
        default="relevance",
        pattern="^(relevance|lastUpdatedDate|submittedDate)$",
    )


class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    source: PaperSource
    arxiv_id: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    title: str
    summary: str
    authors: list[str]
    published_at: datetime | None = None
    updated_at: datetime | None = None
    landing_page_url: HttpUrl
    pdf_url: HttpUrl | None


class PaperSearchAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: PaperSource
    status: Literal["completed", "empty", "failed"]
    result_count: int = Field(default=0, ge=0)
    error_category: str | None = None
    error_message: str | None = None


class PaperSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    papers: list[Paper]
    provider: PaperSource | None
    attempts: list[PaperSearchAttempt]


class PdfDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_url: HttpUrl
    page_count: int = Field(ge=0)
    extracted_pages: int = Field(ge=0)
    extracted_characters: int = Field(ge=0)
    text: str
    truncated: bool = False
    extraction_warnings: list[str] = Field(default_factory=list)
