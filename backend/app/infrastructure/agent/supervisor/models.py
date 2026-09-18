import re
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from app.domain.papers import Paper, PaperSource
from app.domain.rag import (
    CoverageMatrix,
    RetrievalMode,
    RetrievalStrategy,
)


class AgentName(StrEnum):
    SEARCH = "search"
    READER = "reader"
    ANALYST = "analyst"
    WRITER = "writer"


class ResearchTaskType(StrEnum):
    """Task shape decided by the research planner before any agent runs."""

    FACT = "fact"
    COMPARISON = "comparison"
    SURVEY = "survey"
    SET_DISCOVERY = "set_discovery"
    OPEN = "open"


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


class ReaderDepth(StrEnum):
    QUICK = "quick"
    DEEP = "deep"


class DecisionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[str] = Field(max_length=6)
    missing_information: list[str] = Field(max_length=4)
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
    depth: ReaderDepth
    source_artifact_id: UUID | None = None
    paper_id: str | None = Field(default=None, min_length=1)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)

    @property
    def target_paper_ids(self) -> list[str]:
        """Explicitly scoped local papers, supporting single and multi-paper reads."""

        scoped = list(dict.fromkeys(self.paper_ids))
        if self.paper_id is not None and self.paper_id not in scoped:
            scoped.insert(0, self.paper_id)
        return scoped

    @model_validator(mode="after")
    def reject_conflicting_paper_scope(self) -> Self:
        if self.paper_ids and self.paper_id is not None:
            raise ValueError("Use paper_ids for multi-paper scope instead of paper_id")
        return self


class ReadingSubQuestion(BaseModel):
    """One planned retrieval sub-question with its own scope, mode and dimension."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=500)
    retrieval_query: str = Field(min_length=2, max_length=2_000)
    mode: RetrievalMode = RetrievalMode.METHOD
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    dimension: str | None = Field(default=None, max_length=100)


class ReadingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_retrieval: bool
    query: str | None = Field(default=None, min_length=2, max_length=2_000)
    mode: RetrievalMode | None = None
    strategy: RetrievalStrategy | None = None
    coverage_dimensions: list[str] = Field(default_factory=list, max_length=6)
    sub_questions: list[ReadingSubQuestion] = Field(default_factory=list, max_length=6)
    evidence_requirements: list[str] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def require_query_when_retrieval_is_needed(self) -> Self:
        if self.needs_retrieval and (self.query is None or self.mode is None):
            raise ValueError("query and mode are required when retrieval is needed")
        if not self.needs_retrieval and (self.query is not None or self.mode is not None):
            raise ValueError("query and mode must be omitted when retrieval is unnecessary")
        if not self.needs_retrieval and self.sub_questions:
            raise ValueError("sub_questions require needs_retrieval=true")
        return self

    def retrieval_queries(self) -> list[ReadingSubQuestion]:
        """Sub-questions to execute; a single-query plan still yields one entry."""

        if self.sub_questions:
            return list(self.sub_questions)
        if self.query is None or self.mode is None:
            return []
        return [
            ReadingSubQuestion(
                question=self.query,
                retrieval_query=self.query,
                mode=self.mode,
            )
        ]


class CoverageGapJudgment(BaseModel):
    """Evidence-backed verdict for one paper x dimension cell.

    ``covered`` means the retrieved material really answers the dimension and must
    name the Evidence Library entries that do it, ``not_stated`` means the paper
    does not address it at all, and ``missing`` means evidence likely exists but
    was not retrieved. Claiming coverage without evidence is rejected, so a cell
    can never be marked answered by assertion alone.
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(min_length=1, max_length=200)
    dimension: str = Field(min_length=1, max_length=100)
    status: Literal["covered", "missing", "not_stated"]
    reason: str = Field(min_length=1, max_length=400)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def require_evidence_for_covered(self) -> Self:
        if self.status == "covered" and not self.evidence_ids:
            raise ValueError(
                "A covered verdict must cite at least one Evidence Library ID"
            )
        if self.status == "missing" and self.evidence_ids:
            raise ValueError("A missing verdict is a retrieval gap, not evidence")
        for evidence_id in self.evidence_ids:
            if re.fullmatch(r"E-[0-9a-f]{12}", evidence_id) is None:
                raise ValueError(f"Invalid Evidence Library ID: {evidence_id}")
        return self


class ReadingGapQuery(BaseModel):
    """One targeted retrieval that repairs a specific evidence gap."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=2, max_length=2_000)
    mode: RetrievalMode = RetrievalMode.METHOD
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    dimension: str | None = Field(default=None, max_length=100)


class ReadingEvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_sufficient: bool
    coverage_summary: str = Field(min_length=1, max_length=1_000)
    covered_requirements: list[str] = Field(max_length=8)
    missing_requirements: list[str] = Field(max_length=8)
    retry_recommended: bool
    next_query: str | None = Field(default=None, min_length=2, max_length=2_000)
    next_mode: RetrievalMode | None = None
    next_queries: list[ReadingGapQuery] = Field(default_factory=list, max_length=4)
    coverage_judgments: list[CoverageGapJudgment] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def require_retry_plan_when_insufficient(self) -> Self:
        if self.evidence_sufficient and self.retry_recommended:
            raise ValueError("retry_recommended must be false when evidence is sufficient")
        has_retry_parameters = (
            self.next_query is not None
            or self.next_mode is not None
            or bool(self.next_queries)
        )
        if self.retry_recommended and (
            (self.next_query is None or self.next_mode is None) and not self.next_queries
        ):
            raise ValueError(
                "next_query and next_mode, or next_queries, are required for a retry"
            )
        if not self.retry_recommended and has_retry_parameters:
            raise ValueError("retry parameters require retry_recommended=true")
        return self

    def gap_queries(self) -> list[ReadingGapQuery]:
        """Normalized repair queries, accepting both the single and list form."""

        if self.next_queries:
            return list(self.next_queries)
        if self.next_query is None or self.next_mode is None:
            return []
        return [ReadingGapQuery(query=self.next_query, mode=self.next_mode)]


class ResearchPlan(BaseModel):
    """Explicit task-level plan produced by AI before the supervisor starts routing."""

    model_config = ConfigDict(extra="forbid")

    task_type: ResearchTaskType
    retrieval_strategy: RetrievalStrategy
    answer_dimensions: list[str] = Field(default_factory=list, max_length=6)
    target_paper_count: int | None = Field(default=None, ge=1, le=50)
    comparison_targets: list[str] = Field(default_factory=list, max_length=20)
    requires_external_search: bool
    requires_local_corpus: bool
    stopping_criteria: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(min_length=1, max_length=500)


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

    evidence_id: str = Field(pattern=r"^E-[0-9a-f]{12}$")
    claim: str
    page: int | None = Field(default=None, ge=1)
    excerpt: str


class ReadingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_or_material: str
    analysis_summary: str = Field(min_length=1, max_length=8_000)
    objective_satisfied: bool
    answered_points: list[str] = Field(max_length=12)
    blocking_gaps: list[str] = Field(max_length=8)
    evidence: list[PaperEvidence]
    evidence_scope: str
    limitations: list[str] = Field(max_length=8)


class PaperAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
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

    paper_id: str
    source: PaperSource
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
    retrieval_strategy: RetrievalStrategy | None = None
    retrieval_hit_count: int = Field(default=0, ge=0)
    retrieval_rounds: int = Field(default=0, ge=0)
    attempted_queries: list[str] = Field(default_factory=list)
    target_paper_ids: list[str] = Field(default_factory=list)
    missing_paper_ids: list[str] = Field(default_factory=list)
    coverage: CoverageMatrix | None = None


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


class CitationIssue(BaseModel):
    """One citation that an answer makes but its evidence does not support."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    kind: Literal["unknown_evidence_id", "unsupported_claim"]
    detail: str = Field(min_length=1, max_length=400)


class CitationVerification(BaseModel):
    """Result of checking every citation in a final answer.

    ``unknown`` citations reference an Evidence ID that does not exist in any
    supplied library, and ``issues`` are citations whose own evidence does not
    support the sentence they are attached to.
    """

    model_config = ConfigDict(extra="forbid")

    checked: int = Field(default=0, ge=0)
    supported: list[str] = Field(default_factory=list)
    issues: list[CitationIssue] = Field(default_factory=list)
    verification_error: str | None = Field(default=None, max_length=400)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_problems(self) -> bool:
        return bool(self.issues) or self.verification_error is not None


class CitationSupportItem(BaseModel):
    """The verifier's verdict for one citation in a final answer."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(pattern=r"^E-[0-9a-f]{12}$")
    supported: bool
    reason: str = Field(min_length=1, max_length=400)


class CitationSupportAssessment(BaseModel):
    """Model output of the citation-support check."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[CitationSupportItem] = Field(default_factory=list, max_length=40)


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
