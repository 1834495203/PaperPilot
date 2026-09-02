from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.papers import Paper
from app.domain.rag import RetrievalMode


class AgentName(StrEnum):
    SEARCH = "search"
    READER = "reader"
    ANALYST = "analyst"
    WRITER = "writer"


class ArtifactKind(StrEnum):
    SEARCH_RESULT = "search_result"
    READING_REPORT = "reading_report"
    ANALYSIS_REPORT = "analysis_report"


class PaperRelevance(StrEnum):
    DIRECT = "direct"
    ADJACENT = "adjacent"
    IRRELEVANT = "irrelevant"


class SearchStatus(StrEnum):
    COMPLETED = "completed"
    NO_MATCHES = "no_matches"
    RATE_LIMITED = "rate_limited"
    PROVIDER_ERROR = "provider_error"
    FAILED = "failed"


class SearchFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    message: str
    retryable: bool
    status_code: int | None = None
    retry_after_seconds: float | None = Field(default=None, ge=0)


class DecisionSource(StrEnum):
    MODEL = "model"
    POLICY = "policy"
    WORKFLOW = "workflow"
    TOOL = "tool"
    EXTERNAL = "external"


class DecisionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[str]
    missing_information: list[str]
    decision_summary: str = Field(min_length=1, max_length=500)


class AgentTaskBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=500)


class SearchTask(AgentTaskBase):
    agent: Literal[AgentName.SEARCH] = AgentName.SEARCH
    query: str = Field(min_length=2)
    prior_search_artifact_ids: list[UUID] = Field(default_factory=list)


class ReaderTask(AgentTaskBase):
    agent: Literal[AgentName.READER] = AgentName.READER
    source_artifact_id: UUID | None = None
    paper_id: str = Field(min_length=1)


class ReadingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_retrieval: bool
    query: str | None = Field(default=None, min_length=2, max_length=2_000)
    mode: RetrievalMode | None = None
    evidence_requirements: list[str] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_query_when_retrieval_is_needed(self) -> Self:
        if self.needs_retrieval and (self.query is None or self.mode is None):
            raise ValueError("query and mode are required when retrieval is needed")
        if not self.needs_retrieval and (self.query is not None or self.mode is not None):
            raise ValueError("query and mode must be omitted when retrieval is unnecessary")
        return self


class ReadingEvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_sufficient: bool
    coverage_summary: str = Field(min_length=1, max_length=1_000)
    covered_requirements: list[str] = Field(max_length=8)
    missing_requirements: list[str] = Field(max_length=8)
    next_query: str | None = Field(default=None, min_length=2, max_length=2_000)
    next_mode: RetrievalMode | None = None

    @model_validator(mode="after")
    def require_retry_plan_when_insufficient(self) -> Self:
        if not self.evidence_sufficient and (
            self.next_query is None or self.next_mode is None
        ):
            raise ValueError(
                "next_query and next_mode are required when evidence is insufficient"
            )
        return self


class AnalystTask(AgentTaskBase):
    agent: Literal[AgentName.ANALYST] = AgentName.ANALYST
    source_artifact_ids: list[UUID] = Field(min_length=1)


class WriterTask(AgentTaskBase):
    agent: Literal[AgentName.WRITER] = AgentName.WRITER
    source_artifact_ids: list[UUID] = Field(default_factory=list)


AgentTask = Annotated[
    SearchTask | ReaderTask | AnalystTask | WriterTask,
    Field(discriminator="agent"),
]


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: DecisionAssessment
    task: AgentTask


class PaperEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    page: int | None = Field(default=None, ge=1)
    excerpt: str


class ReadingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_or_material: str
    analysis_summary: str = Field(min_length=1, max_length=1_000)
    answer_material: str = Field(min_length=1, max_length=8_000)
    objective_satisfied: bool
    answered_points: list[str] = Field(max_length=12)
    blocking_gaps: list[str] = Field(max_length=8)
    evidence: list[PaperEvidence]
    evidence_scope: str
    limitations: list[str] = Field(max_length=8)


class PaperAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    relevance: PaperRelevance
    relevance_reason: str
    matched_topics: list[str]


class SearchScreening(BaseModel):
    model_config = ConfigDict(extra="forbid")

    screening_summary: str
    assessments: list[PaperAssessment]
    continue_search: bool
    rewritten_query: str | None = Field(default=None, min_length=2, max_length=300)

    @model_validator(mode="after")
    def require_query_when_continuing(self) -> Self:
        if self.continue_search and self.rewritten_query is None:
            raise ValueError("rewritten_query is required when continue_search is true")
        return self


class SearchReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempted_queries: list[str]
    status: SearchStatus = SearchStatus.COMPLETED
    failures: list[SearchFailure] = Field(default_factory=list)
    papers: list[Paper]
    assessments: list[PaperAssessment]
    screening_summary: str


class AnalysisFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding: str
    evidence: list[str]
    confidence: Literal["low", "medium", "high"]
    limitation: str | None = None


class AnalysisReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_type: str
    analysis_summary: str
    comparison_dimensions: list[str]
    findings: list[AnalysisFinding]
    research_gaps: list[str]
    novelty_assessment: str
    unresolved_questions: list[str]


class SearchPaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    title: str
    relevance: PaperRelevance
    relevance_reason: str
    matched_topics: list[str]


class SearchAgentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[ArtifactKind.SEARCH_RESULT] = ArtifactKind.SEARCH_RESULT
    summary: str = Field(max_length=1_000)
    status: SearchStatus = SearchStatus.COMPLETED
    failures: list[SearchFailure] = Field(default_factory=list)
    attempted_queries: list[str]
    direct_count: int = Field(ge=0)
    adjacent_count: int = Field(ge=0)
    irrelevant_count: int = Field(ge=0)
    papers: list[SearchPaperSummary]


class ReaderAgentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[ArtifactKind.READING_REPORT] = ArtifactKind.READING_REPORT
    summary: str = Field(max_length=1_000)
    paper_or_material: str
    objective_satisfied: bool
    answered_points: list[str]
    blocking_gaps: list[str]
    evidence_scope: str
    limitations: list[str]
    pdf_truncated: bool
    pdf_error: str | None = None
    retrieval_mode: RetrievalMode | None = None
    retrieval_hit_count: int = Field(default=0, ge=0)
    retrieval_rounds: int = Field(default=0, ge=0)
    attempted_queries: list[str] = Field(default_factory=list)


class AnalystAgentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[ArtifactKind.ANALYSIS_REPORT] = ArtifactKind.ANALYSIS_REPORT
    summary: str = Field(max_length=1_000)
    finding_count: int = Field(ge=0)
    research_gap_count: int = Field(ge=0)
    novelty_assessment: str
    research_gaps: list[str]
    unresolved_questions: list[str]


AgentSupervisorSummary = Annotated[
    SearchAgentSummary | ReaderAgentSummary | AnalystAgentSummary,
    Field(discriminator="kind"),
]


class AgentArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    title: str
    supervisor_summary: AgentSupervisorSummary
    content: str
    source_artifact_ids: list[UUID] = Field(default_factory=list)

    @property
    def kind(self) -> ArtifactKind:
        return self.supervisor_summary.kind


class CompletedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentName
    objective: str
    artifact_id: UUID | None = None
