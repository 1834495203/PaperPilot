import json
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, create_autospec
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from app.domain.entities import Message
from app.domain.enums import MessageRole
from app.domain.papers import Paper, PaperSearchAttempt, PaperSearchResult, PaperSource
from app.domain.ports import PaperSearchGateway
from app.infrastructure.agent.message_mapper import LangChainMessageMapper
from app.infrastructure.agent.search.tools import AcademicPaperSearchAgentTool
from app.infrastructure.agent.supervisor.model_gateway import ChatModelGateway


class ExampleStructuredOutput(BaseModel):
    answer: str
    confidence: int


class StubJsonModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = iter(responses)
        self.calls: list[list[BaseMessage]] = []
        self.bind_options: dict[str, object] = {}
        self.bound_tools: list[object] = []

    def bind(self, **kwargs: object) -> "StubJsonModel":
        self.bind_options = kwargs
        return self

    def bind_tools(self, tools: list[object]) -> "StubJsonModel":
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(messages)
        return next(self.responses)


@pytest.mark.asyncio
async def test_structured_generation_uses_json_mode_and_specific_repair_feedback() -> None:
    model = StubJsonModel(
        [
            AIMessage(content='{"answer":"first" "confidence":1}'),
            AIMessage(content='{"answer":"repaired","confidence":1}'),
        ]
    )
    gateway = ChatModelGateway(cast(BaseChatModel, model))

    result = await gateway.generate_structured(
        [HumanMessage(content="Answer the question")],
        ExampleStructuredOutput,
    )

    assert result.value.answer == "repaired"
    assert model.bind_options == {"response_format": {"type": "json_object"}}
    repair_prompt = str(model.calls[1][-1].content)
    assert "json_invalid" in repair_prompt
    assert "Rebuild the entire object from scratch" in repair_prompt


@pytest.mark.asyncio
async def test_tool_generation_retries_with_argument_validation_reason() -> None:
    model = StubJsonModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_academic_papers",
                        "args": {
                            "query": 42,
                            "max_results": 3,
                            "sort_by": "relevance",
                            "decision_summary": "Find evidence",
                        },
                        "id": "bad-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_academic_papers",
                        "args": {
                            "query": "agentic RAG",
                            "max_results": 3,
                            "sort_by": "relevance",
                            "decision_summary": "Find evidence",
                        },
                        "id": "repaired-call",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    gateway = ChatModelGateway(cast(BaseChatModel, model))
    paper_gateway = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    tool = AcademicPaperSearchAgentTool(paper_gateway).as_langchain_tool()

    result = await gateway.generate_tool_call(
        [HumanMessage(content="Search for a paper")],
        [tool],
    )

    assert result.call_id == "repaired-call"
    assert result.arguments["query"] == "agentic RAG"
    repair_prompt = str(model.calls[1][-1].content)
    assert "string_type" in repair_prompt
    assert "Generate exactly one new tool call" in repair_prompt


@pytest.mark.asyncio
async def test_academic_agent_tool_uses_gateway_and_returns_explicit_result() -> None:
    gateway = cast(
        PaperSearchGateway,
        create_autospec(PaperSearchGateway, instance=True),
    )
    paper = Paper.model_validate(
        {
            "paper_id": "openalex:W1",
            "source": "openalex",
            "arxiv_id": "2401.00001",
            "external_ids": {"openalex": "W1", "arxiv": "2401.00001"},
            "title": "A Typed Agent Architecture",
            "summary": "An example paper.",
            "authors": ["Ada Example"],
            "published_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "landing_page_url": "https://openalex.org/W1",
            "pdf_url": "https://arxiv.org/pdf/2401.00001",
        }
    )
    search = cast(AsyncMock, gateway.search)
    search.return_value = PaperSearchResult(
        papers=[paper],
        provider=PaperSource.OPENALEX,
        attempts=[
            PaperSearchAttempt(
                provider=PaperSource.OPENALEX,
                status="completed",
                result_count=1,
            )
        ],
    )
    tool = AcademicPaperSearchAgentTool(gateway)

    result = await tool.execute(
        {
            "query": "typed agents",
            "max_results": 3,
            "sort_by": "relevance",
            "decision_summary": "This query targets typed agent architectures",
        }
    )

    search.assert_awaited_once()
    assert json.loads(result.content)[0]["paper_id"] == "openalex:W1"
    assert result.event_payload["result_count"] == 1
    assert result.persisted_summary == {
        "result_count": 1,
        "provider": "openalex",
        "attempts": [
            {
                "provider": "openalex",
                "status": "completed",
                "result_count": 1,
                "error_category": None,
                "error_message": None,
            }
        ],
        "paper_ids": ["openalex:W1"],
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
                    "name": "search_academic_papers",
                    "args": {"query": "agent architecture"},
                    "type": "tool_call",
                }
            ]
        },
    )

    mapped = mapper.to_langchain([message])

    assert len(mapped) == 1
    assert isinstance(mapped[0], AIMessage)
    assert mapped[0].tool_calls[0]["name"] == "search_academic_papers"
    assert mapped[0].tool_calls[0]["args"] == {"query": "agent architecture"}
