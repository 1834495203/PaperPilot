from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from app.application.agent import AgentRunContext
from app.infrastructure.agent.supervisor.analyst_agent import AnalystAgentNode
from app.infrastructure.agent.supervisor.models import (
    AgentName,
    DecisionAssessment,
    ReaderDepth,
    SupervisorDecision,
    WriterTask,
)
from app.infrastructure.agent.supervisor.planner_agent import ResearchPlannerNode
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.search_agent import SearchAgentNode
from app.infrastructure.agent.supervisor.state import SupervisorState, SupervisorStateUpdate
from app.infrastructure.agent.supervisor.supervisor_node import SupervisorNode
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode

RouteName = Literal["search", "reader", "analyst", "writer"]
ReaderExitRoute = Literal["supervisor", "reader_exit"]


class SupervisorGraphBuilder:
    def __init__(
        self,
        *,
        supervisor: SupervisorNode,
        search: SearchAgentNode,
        reader: ReaderAgentNode,
        analyst: AnalystAgentNode,
        writer: WriterAgentNode,
        planner: ResearchPlannerNode | None = None,
    ) -> None:
        self._supervisor = supervisor
        self._search = search
        self._reader = reader
        self._analyst = analyst
        self._writer = writer
        self._planner = planner

    def build(
        self,
    ) -> CompiledStateGraph[
        SupervisorState,
        AgentRunContext,
        SupervisorState,
        SupervisorState,
    ]:
        builder = StateGraph(SupervisorState, context_schema=AgentRunContext)
        builder.add_node("supervisor", self._supervisor)
        builder.add_node("search", self._search)
        builder.add_node("reader", self._reader)
        builder.add_node("analyst", self._analyst)
        builder.add_node("writer", self._writer)
        if self._planner is None:
            builder.add_edge(START, "supervisor")
        else:
            builder.add_node("planner", self._planner)
            builder.add_edge(START, "planner")
            builder.add_edge("planner", "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._route_supervisor,
            {
                AgentName.SEARCH.value: "search",
                AgentName.READER.value: "reader",
                AgentName.ANALYST.value: "analyst",
                AgentName.WRITER.value: "writer",
            },
        )
        builder.add_edge("search", "supervisor")
        builder.add_conditional_edges(
            "reader",
            self._route_after_reader,
            {"supervisor": "supervisor", "reader_exit": "reader_exit"},
        )
        builder.add_node("reader_exit", self._reader_exit)
        builder.add_edge("reader_exit", "writer")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", END)
        return builder.compile()

    @staticmethod
    def _route_supervisor(state: SupervisorState) -> RouteName:
        decision = state["decision"]
        if decision is None:
            raise ValueError("Supervisor must produce a routing decision")
        return decision.task.agent.value

    @staticmethod
    def _route_after_reader(state: SupervisorState) -> ReaderExitRoute:
        """A finished quick read answers one narrow question and is done."""

        outcome = state["reader_outcome"]
        if outcome is not None and outcome.depth is ReaderDepth.QUICK:
            return "reader_exit"
        return "supervisor"

    @staticmethod
    async def _reader_exit(
        state: SupervisorState,
        runtime: Runtime[AgentRunContext],
    ) -> SupervisorStateUpdate:
        """Workflow policy: hand a completed quick read straight to Writer.

        This runs instead of Supervisor, so the quick path costs no extra routing
        model call.
        """

        del runtime
        outcome = state["reader_outcome"]
        if outcome is None:
            raise ValueError("Reader exit requires a finished reader outcome")
        return {
            "decision": SupervisorDecision(
                assessment=DecisionAssessment(
                    observations=["Quick Reader completed its single scoped evidence pass"],
                    missing_information=outcome.missing_requirements,
                    decision_summary=(
                        "The quick path is complete; answer the exact question from its evidence"
                    ),
                ),
                task=WriterTask(
                    objective=(
                        "Answer the user's exact question directly and concisely using the Reader "
                        "artifact; state only material evidence limitations"
                    ),
                    source_artifact_ids=[outcome.artifact_id],
                ),
            )
        }
