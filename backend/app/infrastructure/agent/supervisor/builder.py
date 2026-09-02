from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.application.agent import AgentRunContext
from app.infrastructure.agent.supervisor.analyst_agent import AnalystAgentNode
from app.infrastructure.agent.supervisor.models import AgentName
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.search_agent import SearchAgentNode
from app.infrastructure.agent.supervisor.state import SupervisorState
from app.infrastructure.agent.supervisor.supervisor_node import SupervisorNode
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode

RouteName = Literal["search", "reader", "analyst", "writer"]


class SupervisorGraphBuilder:
    def __init__(
        self,
        *,
        supervisor: SupervisorNode,
        search: SearchAgentNode,
        reader: ReaderAgentNode,
        analyst: AnalystAgentNode,
        writer: WriterAgentNode,
    ) -> None:
        self._supervisor = supervisor
        self._search = search
        self._reader = reader
        self._analyst = analyst
        self._writer = writer

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
        builder.add_edge(START, "supervisor")
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
        builder.add_edge("reader", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", END)
        return builder.compile()

    @staticmethod
    def _route_supervisor(state: SupervisorState) -> RouteName:
        decision = state["decision"]
        if decision is None:
            raise ValueError("Supervisor must produce a routing decision")
        return decision.task.agent.value
