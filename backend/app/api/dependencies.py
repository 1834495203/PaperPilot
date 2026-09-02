from typing import cast

from fastapi import Request

from app.application.chat_service import ChatService
from app.application.paper_library import PaperLibraryService


def get_chat_service(request: Request) -> ChatService:
    return cast(ChatService, request.app.state.container.chat_service)


def get_paper_library(request: Request) -> PaperLibraryService:
    return cast(PaperLibraryService, request.app.state.container.paper_library)
