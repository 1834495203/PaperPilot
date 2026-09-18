from langchain_core.language_models.chat_models import BaseChatModel

from app.application.paper_library import PaperLibraryService
from app.application.tree_retrieval import TreeRagRetriever
from app.domain.ports import ConversationStore, PaperDocumentGateway, PaperSearchGateway
from app.infrastructure.agent.recording import AgentExecutionRecorder
from app.infrastructure.agent.search.tools import AcademicPaperSearchAgentTool
from app.infrastructure.agent.supervisor.analyst_agent import AnalystAgentNode
from app.infrastructure.agent.supervisor.builder import SupervisorGraphBuilder
from app.infrastructure.agent.supervisor.graph import SupervisorAgentGraph
from app.infrastructure.agent.supervisor.model_gateway import ChatModelGateway
from app.infrastructure.agent.supervisor.planner_agent import ResearchPlannerNode
from app.infrastructure.agent.supervisor.reader_agent import ReaderAgentNode
from app.infrastructure.agent.supervisor.search_agent import SearchAgentNode
from app.infrastructure.agent.supervisor.supervisor_node import SupervisorNode
from app.infrastructure.agent.supervisor.writer_agent import WriterAgentNode


def create_supervisor_agent(
    *,
    model: BaseChatModel,
    paper_search: PaperSearchGateway,
    paper_document: PaperDocumentGateway,
    store: ConversationStore,
    max_steps: int,
    search_max_iterations: int,
    reader_max_retrieval_rounds: int,
    paper_retriever: TreeRagRetriever | None = None,
    paper_library: PaperLibraryService | None = None,
    enable_research_planner: bool = True,
    planner_max_answer_dimensions: int = 4,
    enable_citation_verification: bool = True,
    citation_verification_max_citations: int = 12,
) -> SupervisorAgentGraph:
    model_gateway = ChatModelGateway(model)
    recorder = AgentExecutionRecorder(store)
    graph = SupervisorGraphBuilder(
        supervisor=SupervisorNode(model_gateway, max_steps=max_steps),
        search=SearchAgentNode(
            model=model_gateway,
            tool=AcademicPaperSearchAgentTool(paper_search),
            recorder=recorder,
            max_iterations=search_max_iterations,
        ),
        reader=ReaderAgentNode(
            model_gateway,
            document_gateway=paper_document,
            paper_retriever=paper_retriever,
            paper_library=paper_library,
            recorder=recorder,
            max_retrieval_rounds=reader_max_retrieval_rounds,
        ),
        analyst=AnalystAgentNode(model_gateway),
        writer=WriterAgentNode(
            model=model_gateway,
            recorder=recorder,
            verify_citations=enable_citation_verification,
            max_verified_citations=citation_verification_max_citations,
        ),
        planner=(
            ResearchPlannerNode(
                model_gateway,
                max_answer_dimensions=planner_max_answer_dimensions,
            )
            if enable_research_planner
            else None
        ),
    ).build()
    return SupervisorAgentGraph(graph)
