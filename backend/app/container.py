from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import create_async_engine

from app.application.chat_service import ChatService
from app.config import Settings
from app.infrastructure.agent.supervisor.factory import create_supervisor_agent
from app.infrastructure.db.store import SqlAlchemyConversationStore
from app.infrastructure.tools.arxiv import ArxivPaperSearchGateway


class ApplicationContainer:
    def __init__(self, settings: Settings) -> None:
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        self.store = SqlAlchemyConversationStore(engine)
        paper_search = ArxivPaperSearchGateway(
            api_url=settings.arxiv_api_url,
            timeout_seconds=settings.arxiv_timeout_seconds,
        )
        api_key = settings.llm_api_key
        if not api_key.get_secret_value():
            api_key = SecretStr("not-configured")
        model = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=api_key,
            base_url=settings.llm_base_url,
            stream_usage=True,
        )
        agent = create_supervisor_agent(
            model=model,
            paper_search=paper_search,
            store=self.store,
            max_steps=settings.supervisor_max_steps,
        )
        self.chat_service = ChatService(self.store, agent)

    async def initialize(self) -> None:
        await self.store.initialize()

    async def close(self) -> None:
        await self.store.close()
