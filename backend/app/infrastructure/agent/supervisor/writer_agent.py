"""Write the final answer and verify that its citations support its claims."""

import json
import re
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
    CitationIssue,
    CitationSupportAssessment,
    CitationVerification,
    CompletedStep,
    DecisionSource,
    WriterTask,
)
from app.infrastructure.agent.supervisor.prompts import (
    CITATION_VERIFICATION_PROMPT,
    WRITER_PROMPT,
)
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import (
    publish_metrics,
    render_artifacts,
    resolve_evidence,
    select_artifacts,
    with_usage,
)


class WriterAgentNode:
    """Final answer plus a citation-support check over the answer it produced."""

    _CITATION_RE = re.compile(r"\[E-([0-9a-f]{12})\]")

    def __init__(
        self,
        *,
        model: AgentModelGateway,
        recorder: AgentExecutionRecorder,
        verify_citations: bool = True,
        max_verified_citations: int = 12,
    ) -> None:
        if max_verified_citations < 1:
            raise ValueError("max_verified_citations must be positive")
        self._model = model
        self._recorder = recorder
        self._citation_verification_enabled = verify_citations
        self._max_verified_citations = max_verified_citations

    @classmethod
    def _extract_citations(
        cls,
        text: str,
        evidence_by_id: dict[str, Evidence],
    ) -> tuple[list[dict[str, JsonValue]], list[str]]:
        """Split cited Evidence IDs into resolvable citations and unknown ones.

        Unknown IDs are reported rather than dropped: silently deleting a marker
        turns an unsupported claim into a claim with no visible citation at all, so
        the reader cannot tell the difference.
        """

        citations: list[dict[str, JsonValue]] = []
        unknown: list[str] = []
        seen: set[str] = set()
        for match in cls._CITATION_RE.finditer(text):
            evidence_id = f"E-{match.group(1)}"
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                unknown.append(evidence_id)
                continue
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
        return citations, unknown

    @staticmethod
    def _unknown_citation_issues(unknown_ids: list[str]) -> list[CitationIssue]:
        return [
            CitationIssue(
                evidence_id=evidence_id,
                kind="unknown_evidence_id",
                detail="The answer cites an Evidence ID that no supplied artifact defines",
            )
            for evidence_id in unknown_ids
        ]

    async def _verify_citations(
        self,
        context: AgentRunContext,
        *,
        answer: str,
        citations: list[dict[str, JsonValue]],
        unknown_ids: list[str],
    ) -> CitationVerification:
        """Ask the model whether each citation supports the claim it annotates."""

        issues = self._unknown_citation_issues(unknown_ids)
        if not self._citation_verification_enabled or not citations:
            return CitationVerification(checked=len(citations), issues=issues)
        checked = citations[: self._max_verified_citations]
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "writer",
                "stage": "writer.citation_check",
                "summary": "Writer 正在核验引用是否支持结论",
                "citation_count": len(checked),
                "unresolved_citation_count": len(unknown_ids),
            },
        )
        result = await self._model.generate_structured(
            [
                SystemMessage(content=CITATION_VERIFICATION_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "answer": answer,
                            "cited_evidence": [
                                {
                                    "evidence_id": item["evidence_id"],
                                    "paper_title": item["paper_title"],
                                    "evidence_text": item["excerpt"],
                                }
                                for item in checked
                            ],
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            CitationSupportAssessment,
        )
        known_ids = {str(item["evidence_id"]) for item in checked}
        supported: list[str] = []
        unsupported: list[CitationIssue] = []
        for assessment in result.value.assessments:
            if assessment.evidence_id not in known_ids:
                continue
            if assessment.supported:
                supported.append(assessment.evidence_id)
            else:
                unsupported.append(
                    CitationIssue(
                        evidence_id=assessment.evidence_id,
                        kind="unsupported_claim",
                        detail=assessment.reason,
                    )
                )
        unjudged = sorted(
            known_ids - set(supported) - {item.evidence_id for item in unsupported}
        )
        issues.extend(
            CitationIssue(
                evidence_id=evidence_id,
                kind="unsupported_claim",
                detail="The verification step returned no verdict for this citation",
            )
            for evidence_id in unjudged
        )
        return CitationVerification(
            checked=len(checked),
            supported=supported,
            issues=[*issues, *unsupported],
        )

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
        # Evidence resolves through source relationships, so a citation stays
        # traceable even when only a downstream report was selected.
        evidence_by_id = resolve_evidence(state["artifacts"], task.source_artifact_ids)
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
        citations, unknown_ids = self._extract_citations(result.text, evidence_by_id)
        try:
            verification = await self._verify_citations(
                context,
                answer=result.text,
                citations=citations,
                unknown_ids=unknown_ids,
            )
        except (RuntimeError, ValueError, TypeError) as error:
            # Verification is a safety net, not a gate: an unavailable checker must
            # not turn a finished answer into a failed run.
            verification = CitationVerification(
                checked=len(citations),
                issues=self._unknown_citation_issues(unknown_ids),
                verification_error=str(error)[:400],
            )
        message_id = await self._recorder.record_assistant_message(
            context=context,
            message=result.message,
            duration_ms=int((perf_counter() - started) * 1000),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
            citations=citations,
            citation_verification=cast(
                JsonValue, verification.model_dump(mode="json")
            ),
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
                "citation_verification": cast(
                    JsonValue, verification.model_dump(mode="json")
                ),
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
