import json

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from app.domain.papers import ArxivSearchInput
from app.domain.ports import PaperSearchGateway
from app.domain.types import JsonValue
from app.infrastructure.agent.tooling import AgentTool, AgentToolResult


class ArxivSearchToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=2, max_length=300)
    max_results: int = Field(default=5, ge=1, le=10)
    sort_by: str = Field(
        default="relevance",
        pattern="^(relevance|lastUpdatedDate|submittedDate)$",
    )
    decision_summary: str = Field(
        min_length=1,
        max_length=500,
        description="Brief reason, in your own words, why this query supports the assigned goal",
    )


class ArxivSearchAgentTool(AgentTool):
    """Search-agent adapter over the domain-level paper search gateway."""

    def __init__(self, gateway: PaperSearchGateway) -> None:
        self._gateway = gateway
        self._tool = StructuredTool.from_function(
            coroutine=self._search_for_model,
            name=self.name,
            description=(
                "Search arXiv for real academic papers relevant to a focused research query."
            ),
            args_schema=ArxivSearchToolInput,
        )

    @property
    def name(self) -> str:
        return "search_arxiv"

    def as_langchain_tool(self) -> BaseTool:
        return self._tool

    async def execute(self, arguments: dict[str, JsonValue]) -> AgentToolResult:
        tool_input = ArxivSearchToolInput.model_validate(arguments)
        search_input = ArxivSearchInput(
            query=tool_input.query,
            max_results=tool_input.max_results,
            sort_by=tool_input.sort_by,
        )
        papers = await self._gateway.search(search_input)
        serialized: list[JsonValue] = [
            paper.model_dump(mode="json") for paper in papers
        ]
        return AgentToolResult(
            content=json.dumps(serialized, ensure_ascii=False),
            result=serialized,
            persisted_summary={
                "result_count": len(papers),
                "paper_ids": [paper.arxiv_id for paper in papers],
                "titles": [paper.title for paper in papers],
            },
            event_payload={"result_count": len(papers), "papers": serialized},
        )

    async def _search_for_model(
        self,
        query: str,
        max_results: int = 5,
        sort_by: str = "relevance",
        decision_summary: str = "Search for evidence relevant to the assigned goal",
    ) -> list[JsonValue]:
        result = await self.execute(
            {
                "query": query,
                "max_results": max_results,
                "sort_by": sort_by,
                "decision_summary": decision_summary,
            }
        )
        if not isinstance(result.result, list):
            raise TypeError("arXiv search result must be a list")
        return result.result
