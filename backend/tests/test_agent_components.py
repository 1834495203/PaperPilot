import json
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from app.domain.entities import Message
from app.domain.enums import MessageRole
from app.domain.papers import Paper
from app.domain.ports import PaperSearchGateway
from app.infrastructure.agent.message_mapper import LangChainMessageMapper
from app.infrastructure.agent.search.tools import ArxivSearchAgentTool


@pytest.mark.asyncio
async def test_arxiv_agent_tool_uses_gateway_and_returns_explicit_result() -> None:
    gateway = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    paper = Paper.model_validate(
        {
            "arxiv_id": "2401.00001",
            "title": "A Typed Agent Architecture",
            "summary": "An example paper.",
            "authors": ["Ada Example"],
            "published_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "abstract_url": "https://arxiv.org/abs/2401.00001",
            "pdf_url": "https://arxiv.org/pdf/2401.00001",
        }
    )
    search = cast(AsyncMock, gateway.search)
    search.return_value = [paper]
    tool = ArxivSearchAgentTool(gateway)

    result = await tool.execute(
        {
            "query": "typed agents",
            "max_results": 3,
            "sort_by": "relevance",
            "decision_summary": "This query targets typed agent architectures",
        }
    )

    search.assert_awaited_once()
    assert json.loads(result.content)[0]["arxiv_id"] == "2401.00001"
    assert result.event_payload["result_count"] == 1
    assert result.persisted_summary == {
        "result_count": 1,
        "paper_ids": ["2401.00001"],
        "titles": ["A Typed Agent Architecture"],
    }


def test_message_mapper_restores_explicit_tool_call_metadata() -> None:
    mapper = LangChainMessageMapper()
    message = Message(
        id=uuid4(),
        conversation_id=uuid4(),
        role=MessageRole.ASSISTANT,
        content="",
        sequence=1,
        created_at=datetime.now(UTC),
        metadata={
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "search_arxiv",
                    "args": {"query": "agent architecture"},
                    "type": "tool_call",
                }
            ]
        },
    )

    mapped = mapper.to_langchain([message])

    assert len(mapped) == 1
    assert isinstance(mapped[0], AIMessage)
    assert mapped[0].tool_calls[0]["name"] == "search_arxiv"
    assert mapped[0].tool_calls[0]["args"] == {"query": "agent architecture"}
