from langchain_core.language_models.chat_models import BaseChatModel

from app.domain.ports import ConversationStore, PaperSearchGateway
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.search.tools import ArxivSearchAgentTool
from app.infrastructure.agent.supervisor.analyst_agent import AnalystAgentNode
from app.infrastructure.agent.supervisor.builder import SupervisorGraphBuilder
from app.infrastructure.agent.supervisor.graph import SupervisorAgentGraph
from app.infrastructure.agent.supervisor.model_gateway import ChatModelGateway
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.search_agent import SearchAgentNode
from app.infrastructure.agent.supervisor.supervisor_node import SupervisorNode
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode


def create_supervisor_agent(
    *,
    model: BaseChatModel,
    paper_search: PaperSearchGateway,
    store: ConversationStore,
    max_steps: int,
) -> SupervisorAgentGraph:
    model_gateway = ChatModelGateway(model)
    recorder = AgentExecutionRecorder(store)
    graph = SupervisorGraphBuilder(
        supervisor=SupervisorNode(model_gateway, max_steps=max_steps),
        search=SearchAgentNode(
            model=model_gateway,
            tool=ArxivSearchAgentTool(paper_search),
            recorder=recorder,
        ),
        reader=ReaderAgentNode(model_gateway),
        analyst=AnalystAgentNode(model_gateway),
        writer=WriterAgentNode(model=model_gateway, recorder=recorder),
    ).build()
    return SupervisorAgentGraph(graph)
