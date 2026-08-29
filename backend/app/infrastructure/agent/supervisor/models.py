from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentName(StrEnum):
    SEARCH = "search"
    READER = "reader"
    ANALYST = "analyst"
    WRITER = "writer"


class ArtifactKind(StrEnum):
    SEARCH_RESULT = "search_result"
    READING_REPORT = "reading_report"
    ANALYSIS_REPORT = "analysis_report"


class DecisionSource(StrEnum):
    MODEL = "model"
    POLICY = "policy"
    WORKFLOW = "workflow"
    TOOL = "tool"
    EXTERNAL = "external"


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    next_agent: AgentName
    observations: list[str]
    missing_information: list[str]
    objective: str = Field(min_length=1, max_length=500)
    decision_summary: str = Field(min_length=1, max_length=500)
    success_criteria: list[str]
    query: str | None = Field(default=None, max_length=300)
    artifact_ids: list[UUID] = Field(default_factory=list)


class AgentArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    kind: ArtifactKind
    title: str
    summary: str
    content: str
    source_artifact_ids: list[UUID] = Field(default_factory=list)


class ReadingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_or_material: str
    analysis_summary: str
    domain: str
    research_problem: str
    motivation: str
    method: str
    underlying_principles: str
    experiments: str
    limitations: list[str]
    evidence: list[str]
    evidence_scope: str


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


class CompletedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentName
    objective: str
    artifact_id: UUID | None = None
