from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ArxivSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=2, max_length=300, description="A focused arXiv search query")
    max_results: int = Field(default=5, ge=1, le=10)
    sort_by: str = Field(
        default="relevance",
        pattern="^(relevance|lastUpdatedDate|submittedDate)$",
    )


class Paper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    published_at: datetime
    updated_at: datetime
    abstract_url: HttpUrl
    pdf_url: HttpUrl | None

