from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.application.agent import AgentRunContext
from app.infrastructure.agent.supervisor.analyst_agent import AnalystAgentNode
from app.infrastructure.agent.supervisor.models import AgentName
from app.infrastructure.agent.supervisor.planner_agent import ResearchPlannerNode
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.search_agent import SearchAgentNode
from app.infrastructure.agent.supervisor.state import SupervisorState
from app.infrastructure.agent.supervisor.supervisor_node import SupervisorNode
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode

RouteName = Literal["search", "reader", "analyst", "writer"]
ReaderExitRoute = Literal["supervisor", "writer"]


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
            {"supervisor": "supervisor", "writer": "writer"},
        )
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
        decision = state["decision"]
        if decision is not None and decision.task.agent is AgentName.WRITER:
            return "writer"
        return "supervisor"
