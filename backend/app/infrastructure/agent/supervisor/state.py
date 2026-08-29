from typing_extensions import TypedDict

from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    CompletedStep,
    SupervisorDecision,
)


class SupervisorState(TypedDict):
    user_request: str
    conversation_context: str
    artifacts: list[AgentArtifact]
    completed_steps: list[CompletedStep]
    decision: SupervisorDecision | None
    step_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int


class SupervisorStateUpdate(TypedDict, total=False):
    artifacts: list[AgentArtifact]
    completed_steps: list[CompletedStep]
    decision: SupervisorDecision | None
    step_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    llm_calls: int
    tool_calls: int
