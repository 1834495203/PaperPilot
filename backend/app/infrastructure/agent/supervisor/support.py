import json
from collections.abc import Sequence
from uuid import UUID

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.rag import Evidence
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
    selected_ids = set(artifact_ids)
    return [artifact for artifact in artifacts if artifact.id in selected_ids]


def resolve_evidence(
    artifacts: Sequence[AgentArtifact],
    artifact_ids: Sequence[UUID],
) -> dict[str, Evidence]:
    """Collect the evidence libraries of the selected artifacts and their sources.

    Writer may receive an Analyst report without the Reader artifacts it was built
    from. Resolving ``source_artifact_ids`` recursively keeps citations traceable in
    that case, instead of dropping every reference as unknown.
    """

    by_id = {artifact.id: artifact for artifact in artifacts}
    visited: set[UUID] = set()
    pending = list(artifact_ids)
    evidence_by_id: dict[str, Evidence] = {}
    while pending:
        artifact_id = pending.pop()
        if artifact_id in visited:
            continue
        visited.add(artifact_id)
        artifact = by_id.get(artifact_id)
        if artifact is None:
            continue
        pending.extend(artifact.source_artifact_ids)
        for evidence in _evidence_from_artifact(artifact):
            evidence_by_id.setdefault(evidence.evidence_id, evidence)
    return evidence_by_id


def _evidence_from_artifact(artifact: AgentArtifact) -> list[Evidence]:
    try:
        payload = json.loads(artifact.content)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict):
        return []
    library = payload.get("evidence_library")
    if not isinstance(library, dict):
        return []
    entries = library.get("evidence")
    if not isinstance(entries, list):
        return []
    evidence: list[Evidence] = []
    for entry in entries:
        try:
            evidence.append(Evidence.model_validate(entry))
        except ValueError:
            continue
    return evidence


def render_artifacts(
    artifacts: Sequence[AgentArtifact],
) -> str:
    return json.dumps(
        [
            {
                "id": str(artifact.id),
                "kind": artifact.kind.value,
                "title": artifact.title,
                "content": artifact.content,
            }
            for artifact in artifacts
        ],
        ensure_ascii=False,
    )


def render_supervisor_context(
    artifacts: Sequence[AgentArtifact],
) -> str:
    """Expose only Agent-authored summaries; detailed reports stay opaque to Supervisor."""

    return json.dumps(
        [
            {
                "id": str(artifact.id),
                "title": artifact.title,
                "summary": artifact.supervisor_summary.model_dump(mode="json"),
            }
            for artifact in artifacts
        ],
        ensure_ascii=False,
    )
