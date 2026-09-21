from typing_extensions import TypedDict

from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    CompletedStep,
    ReaderOutcome,
    ResearchPlan,
    SupervisorDecision,
)


class SupervisorState(TypedDict):
    user_request: str
    conversation_context: str
    research_plan: ResearchPlan | None
    artifacts: list[AgentArtifact]
    completed_steps: list[CompletedStep]
    decision: SupervisorDecision | None
    reader_outcome: ReaderOutcome | None
    step_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int


class SupervisorStateUpdate(TypedDict, total=False):
    research_plan: ResearchPlan | None
    artifacts: list[AgentArtifact]
    completed_steps: list[CompletedStep]
    decision: SupervisorDecision | None
    reader_outcome: ReaderOutcome | None
    step_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int
