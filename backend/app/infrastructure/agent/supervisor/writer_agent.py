import json
import re
from collections.abc import Sequence
from time import perf_counter
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.domain.enums import EventType, MessageRole
from app.domain.rag import Evidence
from app.domain.types import JsonValue
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway
from app.infrastructure.agent.supervisor.models import (
    AgentArtifact,
    CompletedStep,
    DecisionSource,
    WriterTask,
)
from app.infrastructure.agent.supervisor.prompts import WRITER_PROMPT
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    render_artifacts,
    select_artifacts,
    with_usage,
)


class WriterAgentNode:
    def __init__(
        self,
        *,
        model: AgentModelGateway,
        recorder: AgentExecutionRecorder,
    ) -> None:
        self._model = model
        self._recorder = recorder

    _CITATION_RE = re.compile(r"\[E-([0-9a-f]{12})\]")

    @staticmethod
    def _evidence_map(artifacts: Sequence[AgentArtifact]) -> dict[str, Evidence]:
        evidence_by_id: dict[str, Evidence] = {}
        for artifact in artifacts:
            try:
                data = json.loads(artifact.content)
            except (json.JSONDecodeError, TypeError):
                continue
            library = data.get("evidence_library")
            if not isinstance(library, dict):
                continue
            for item in library.get("evidence", []):
                try:
                    evidence = Evidence.model_validate(item)
                except Exception:
                    continue
                evidence_by_id[evidence.evidence_id] = evidence
        return evidence_by_id

    @classmethod
    def _extract_citations(
        cls,
        text: str,
        evidence_by_id: dict[str, Evidence],
    ) -> list[dict[str, JsonValue]]:
        citations: list[dict[str, JsonValue]] = []
        seen: set[str] = set()
        for match in cls._CITATION_RE.finditer(text):
            evidence_id = f"E-{match.group(1)}"
            if evidence_id in seen:
                continue
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                continue
            seen.add(evidence_id)
            citations.append(
                {
                    "evidence_id": evidence_id,
                    "paper_id": evidence.paper_id,
                    "paper_title": evidence.paper_title,
                    "page_start": evidence.page_start,
                    "page_end": evidence.page_end,
                    "excerpt": evidence.evidence_text[:280],
                    "spans": cast(
                        JsonValue,
                        [span.model_dump(mode="json") for span in evidence.spans],
                    ),
                }
            )
        return citations

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        decision = state["decision"]
        if decision is None or not isinstance(decision.task, WriterTask):
            raise ValueError("Writer Agent requires a writer decision")
        task = decision.task
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "writer",
                "stage": "writer",
                "summary": "Writer Agent 节点开始执行",
                "objective": task.objective,
            },
        )
        selected = select_artifacts(state["artifacts"], task.source_artifact_ids)
        evidence_by_id = self._evidence_map(selected)
        prompt = (
            f"User request:\n{state['user_request']}\n\n"
            f"Conversation context:\n{state['conversation_context']}\n\n"
            f"Writing objective:\n{task.objective}\n\n"
            f"Available research artifacts:\n"
            f"{render_artifacts(selected)}"
        )

        async def publish_token(text: str) -> None:
            await context.publisher.publish(
                EventType.TOKEN.value,
                {
                    "source": DecisionSource.MODEL.value,
                    "actor": "writer",
                    "text": text,
                    "stage": "writer",
                },
            )

        started = perf_counter()
        result = await self._model.generate_text(
            [SystemMessage(content=WRITER_PROMPT), HumanMessage(content=prompt)],
            token_consumer=publish_token,
        )
        if not result.text.strip():
            raise RuntimeError("Writer Agent returned an empty response")
        citations = self._extract_citations(result.text, evidence_by_id)
        message_id = await self._recorder.record_assistant_message(
            context=context,
            message=result.message,
            duration_ms=int((perf_counter() - started) * 1000),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
            citations=citations,
        )
        await context.publisher.publish(
            EventType.MESSAGE_COMPLETED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "writer",
                "message_id": str(message_id),
                "role": MessageRole.ASSISTANT.value,
                "content": result.text,
                "citations": cast(JsonValue, citations),
                "has_tool_calls": False,
                "stage": "writer",
            },
        )
        update: SupervisorStateUpdate = {
            "completed_steps": [
                *state["completed_steps"],
                CompletedStep(
                    agent=task.agent,
                    objective=task.objective,
                ),
            ],
            **with_usage(state, result.usage),
        }
        await publish_metrics(context, update)
        return update
