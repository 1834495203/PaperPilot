from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.entities import (
    AgentEvent,
    AgentRun,
    Conversation,
    ConversationMetrics,
    Message,
)
from app.domain.enums import MessageRole
from app.domain.rag import (
    IndexedPaper,
    IndexedPaperDetail,
    RetrievalHit,
    RetrievalMode,
    TreeRetrievalReport,
)
from app.domain.types import JsonValue


class CreateConversationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(default="New research", min_length=1, max_length=200)


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    content: str = Field(min_length=1, max_length=10_000)


class IndexedPaperResponse(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    abstract: str | None
    keywords: list[str]
    doi: str | None
    arxiv_id: str | None
    original_filename: str
    page_count: int
    section_count: int
    node_count: int
    chunk_count: int
    created_at: datetime

    @classmethod
    def from_domain(cls, paper: IndexedPaper) -> "IndexedPaperResponse":
        return cls(
            paper_id=paper.paper_id,
            title=paper.metadata.title,
            authors=paper.metadata.authors,
            abstract=paper.metadata.abstract,
            keywords=paper.metadata.keywords,
            doi=paper.metadata.doi,
            arxiv_id=paper.metadata.arxiv_id,
            original_filename=paper.original_filename,
            page_count=paper.page_count,
            section_count=paper.section_count,
            node_count=paper.node_count,
            chunk_count=paper.chunk_count,
            created_at=paper.created_at,
        )


class PaperTreeNodeResponse(BaseModel):
    node_id: str
    node_type: str
    title: str
    parent_id: str | None
    children_ids: list[str]
    level: int
    section_path: list[str]
    semantic_role: str | None
    block_types: list[str]
    object_labels: list[str]
    page_start: int | None
    page_end: int | None
    text_preview: str
    text: str


class IndexedPaperDetailResponse(BaseModel):
    paper: IndexedPaperResponse
    nodes: list[PaperTreeNodeResponse]

    @classmethod
    def from_domain(cls, detail: IndexedPaperDetail) -> "IndexedPaperDetailResponse":
        return cls(
            paper=IndexedPaperResponse.from_domain(detail.paper),
            nodes=[
                PaperTreeNodeResponse(**node.model_dump(mode="json"))
                for node in detail.nodes
            ],
        )


class RetrievePaperRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2_000)
    paper_ids: list[str] | None = Field(default=None, max_length=20)
    mode: RetrievalMode = RetrievalMode.FACT


class RetrievalHitResponse(BaseModel):
    rank: int
    node_id: str
    paper_id: str
    section_path: list[str]
    semantic_role: str | None
    block_types: list[str]
    object_labels: list[str]
    page_start: int | None
    page_end: int | None
    text: str
    vector_score: float
    ranking_score: float
    source: str
    expanded_from: str | None

    @classmethod
    def from_domain(cls, hit: RetrievalHit) -> "RetrievalHitResponse":
        return cls(**hit.model_dump(mode="json"))


class TreeRetrievalResponse(BaseModel):
    query: str
    mode: RetrievalMode
    paper_ids: list[str]
    searched_globally: bool
    candidate_paper_ids: list[str]
    initial_hit_count: int
    expanded_candidate_count: int
    hits: list[RetrievalHitResponse]

    @classmethod
    def from_domain(cls, report: TreeRetrievalReport) -> "TreeRetrievalResponse":
        return cls(
            query=report.query,
            mode=report.mode,
            paper_ids=report.paper_ids,
            searched_globally=report.searched_globally,
            candidate_paper_ids=report.candidate_paper_ids,
            initial_hit_count=report.initial_hit_count,
            expanded_candidate_count=report.expanded_candidate_count,
            hits=[RetrievalHitResponse.from_domain(hit) for hit in report.hits],
        )


class ConversationResponse(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_domain(cls, conversation: Conversation) -> "ConversationResponse":
        return cls(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    sequence: int
    created_at: datetime
    metadata: dict[str, JsonValue]

    @classmethod
    def from_domain(cls, message: Message) -> "MessageResponse":
        return cls(
            id=message.id,
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            sequence=message.sequence,
            created_at=message.created_at,
            metadata=message.metadata,
        )


class ConversationMetricsResponse(BaseModel):
    conversation_id: UUID
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int
    total_duration_ms: int
    run_count: int
    updated_at: datetime | None

    @classmethod
    def from_domain(cls, metrics: ConversationMetrics) -> "ConversationMetricsResponse":
        return cls(
            conversation_id=metrics.conversation_id,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            llm_calls=metrics.llm_calls,
            tool_calls=metrics.tool_calls,
            total_duration_ms=metrics.total_duration_ms,
            run_count=metrics.run_count,
            updated_at=metrics.updated_at,
        )


class AgentRunResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    status: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int
    duration_ms: int
    error: str | None
    started_at: datetime
    completed_at: datetime | None

    @classmethod
    def from_domain(cls, run: AgentRun) -> "AgentRunResponse":
        return cls(
            id=run.id,
            conversation_id=run.conversation_id,
            status=run.status.value,
            input_tokens=run.metrics.input_tokens,
            output_tokens=run.metrics.output_tokens,
            total_tokens=run.metrics.total_tokens,
            llm_calls=run.metrics.llm_calls,
            tool_calls=run.metrics.tool_calls,
            duration_ms=run.metrics.duration_ms,
            error=run.error,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )


class EventResponse(BaseModel):
    id: UUID
    run_id: UUID
    conversation_id: UUID
    sequence: int
    type: str
    timestamp: datetime
    payload: dict[str, JsonValue]

    @classmethod
    def from_domain(cls, event: AgentEvent) -> "EventResponse":
        return cls(
            id=event.id,
            run_id=event.run_id,
            conversation_id=event.conversation_id,
            sequence=event.sequence,
            type=event.type.value,
            timestamp=event.timestamp,
            payload=event.payload,
        )
