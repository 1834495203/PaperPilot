from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import conversations, health, papers
from app.config import get_settings
from app.container import ApplicationContainer


def create_app() -> FastAPI:
    settings = get_settings()
    container = ApplicationContainer(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.container = container
        await container.initialize()
        try:
            yield
        finally:
            await container.close()

    application = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(conversations.router, prefix=settings.api_prefix)
    application.include_router(papers.router, prefix=settings.api_prefix)
    return application


app = create_app()
