import json
from collections.abc import Sequence
from uuid import UUID

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.infrastructure.agent.supervisor.model_gateway import ModelUsage
from app.infrastructure.agent.supervisor.models import AgentArtifact, DecisionSource
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate


def with_usage(
    state: SupervisorState,
    usage: ModelUsage,
    *,
    llm_calls: int = 1,
    tool_calls: int = 0,
) -> SupervisorStateUpdate:
    return {
        "input_tokens": state["input_tokens"] + usage.input_tokens,
        "output_tokens": state["output_tokens"] + usage.output_tokens,
        "total_tokens": state["total_tokens"] + usage.total_tokens,
        "llm_calls": state["llm_calls"] + llm_calls,
        "tool_calls": state["tool_calls"] + tool_calls,
    }


async def publish_metrics(
    context: AgentRunContext,
    update: SupervisorStateUpdate,
) -> None:
    await context.publisher.publish(
        EventType.METRICS_UPDATED.value,
        {
            "source": DecisionSource.WORKFLOW.value,
            "actor": "metrics",
            "input_tokens": update.get("input_tokens", 0),
            "output_tokens": update.get("output_tokens", 0),
            "total_tokens": update.get("total_tokens", 0),
            "llm_calls": update.get("llm_calls", 0),
            "tool_calls": update.get("tool_calls", 0),
        },
    )


def select_artifacts(
    artifacts: Sequence[AgentArtifact],
    artifact_ids: Sequence[UUID],
) -> list[AgentArtifact]:
    if not artifact_ids:
        return list(artifacts)
    selected_ids = set(artifact_ids)
    return [artifact for artifact in artifacts if artifact.id in selected_ids]


def render_artifacts(
    artifacts: Sequence[AgentArtifact],
    *,
    max_content_chars: int = 24_000,
) -> str:
    remaining = max_content_chars
    rendered: list[dict[str, object]] = []
    for artifact in artifacts:
        if remaining <= 0:
            break
        content = artifact.content[:remaining]
        remaining -= len(content)
        rendered.append(
            {
                "id": str(artifact.id),
                "kind": artifact.kind.value,
                "title": artifact.title,
                "summary": artifact.summary,
                "content": content,
            }
        )
    return json.dumps(rendered, ensure_ascii=False)
