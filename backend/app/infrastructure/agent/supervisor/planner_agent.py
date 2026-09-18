import json
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.domain.enums import EventType
from app.domain.types import JsonValue
from app.infrastructure.agent.supervisor.model_gateway import AgentModelGateway
from app.infrastructure.agent.supervisor.models import DecisionSource, ResearchPlan
from app.infrastructure.agent.supervisor.prompts import RESEARCH_PLANNER_PROMPT
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.support import publish_metrics, with_usage


class ResearchPlannerNode:
    """Model-authored task plan produced once, before the supervisor routes anything.

    The plan states the task type, the retrieval strategy, the dimensions a complete
    answer needs and the stopping criteria. The supervisor reads it as the task
    strategy; the Reader receives it as coverage-dimension guidance.
    """

    def __init__(
        self,
        model: AgentModelGateway,
        *,
        max_answer_dimensions: int = 6,
    ) -> None:
        if max_answer_dimensions < 1:
            raise ValueError("max_answer_dimensions must be positive")
        self._model = model
        self._max_answer_dimensions = max_answer_dimensions

    async def __call__(
        self,
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        context = runtime.context
        await context.publisher.publish(
            EventType.STAGE_STARTED.value,
            {
                "source": DecisionSource.WORKFLOW.value,
                "actor": "planner",
                "stage": "planner",
                "summary": "Planner 正在为本次请求制定显式任务计划",
                "local_corpus_available": context.local_corpus_available,
                "indexed_paper_count": len(context.paper_ids),
            },
        )
        result = await self._model.generate_structured(
            [
                SystemMessage(content=RESEARCH_PLANNER_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        {
                            "user_request": state["user_request"],
                            "conversation_context": state["conversation_context"],
                            "local_corpus_available": context.local_corpus_available,
                            "indexed_paper_ids": list(context.paper_ids),
                            "max_answer_dimensions": self._max_answer_dimensions,
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            ResearchPlan,
        )
        plan = result.value
        await context.publisher.publish(
            EventType.DECISION_RECORDED.value,
            {
                "source": DecisionSource.MODEL.value,
                "actor": "planner",
                "stage": "planner",
                "summary": plan.rationale,
                "task_type": plan.task_type.value,
                "retrieval_strategy": plan.retrieval_strategy.value,
                "answer_dimensions": cast(JsonValue, plan.answer_dimensions),
                "target_paper_count": plan.target_paper_count,
                "comparison_targets": cast(JsonValue, plan.comparison_targets),
                "requires_external_search": plan.requires_external_search,
                "requires_local_corpus": plan.requires_local_corpus,
                "stopping_criteria": cast(JsonValue, plan.stopping_criteria),
            },
        )
        update: SupervisorStateUpdate = {
            "research_plan": plan,
            **with_usage(state, result.usage),
        }
        await publish_metrics(context, update)
        return update
