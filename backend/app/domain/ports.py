from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from app.domain.entities import (
    AgentEvent,
    AgentRun,
    Conversation,
    ConversationMetrics,
    Message,
    RunMetrics,
)
from app.domain.enums import MessageRole, RunStatus
from app.domain.papers import PaperSearchInput, PaperSearchResult, PdfDocument
from app.domain.rag import (
    IndexedTreeNode,
    KeywordSearchResult,
    ParsedPaperDocument,
    TreeIndexNode,
    TreeNodeType,
    TreeVectorMatch,
)
from app.domain.types import JsonValue


class ConversationStore(ABC):
    @abstractmethod
    async def create_conversation(self, title: str) -> Conversation: ...

    @abstractmethod
    async def list_conversations(self) -> Sequence[Conversation]: ...

    @abstractmethod
    async def get_conversation(self, conversation_id: UUID) -> Conversation | None: ...

    @abstractmethod
    async def delete_conversation(self, conversation_id: UUID) -> bool: ...

    @abstractmethod
    async def list_messages(
        self,
        conversation_id: UUID,
        *,
        include_superseded: bool = False,
    ) -> Sequence[Message]:
        """Thread messages; superseded versions are excluded unless requested."""

    @abstractmethod
    async def get_message(
        self,
        conversation_id: UUID,
        message_id: UUID,
    ) -> Message | None: ...

    @abstractmethod
    async def list_events(
        self,
        conversation_id: UUID,
        limit: int,
    ) -> Sequence[AgentEvent]: ...

    @abstractmethod
    async def append_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        metadata: dict[str, JsonValue] | None = None,
    ) -> Message: ...

    @abstractmethod
    async def create_run(self, conversation_id: UUID) -> AgentRun: ...

    @abstractmethod
    async def list_runs(self, conversation_id: UUID) -> Sequence[AgentRun]: ...

    @abstractmethod
    async def get_run(self, conversation_id: UUID, run_id: UUID) -> AgentRun | None: ...

    @abstractmethod
    async def list_run_events(
        self,
        run_id: UUID,
        *,
        after_sequence: int = 0,
    ) -> Sequence[AgentEvent]:
        """Persisted events of one run, for replay after a reconnect."""

    @abstractmethod
    async def finish_run(
        self,
        run_id: UUID,
        status: RunStatus,
        metrics: RunMetrics,
        error: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_conversation_metrics(
        self,
        conversation_id: UUID,
    ) -> ConversationMetrics: ...

    @abstractmethod
    async def refresh_conversation_metrics(
        self,
        conversation_id: UUID,
    ) -> ConversationMetrics: ...

    @abstractmethod
    async def append_tool_call(
        self,
        *,
        run_id: UUID,
        message_id: UUID | None,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, JsonValue],
        result: JsonValue | None,
        error: str | None,
        duration_ms: int | None,
    ) -> None: ...

    @abstractmethod
    async def append_event(self, event: AgentEvent) -> None: ...

    @abstractmethod
    async def supersede_messages_from(
        self,
        conversation_id: UUID,
        sequence: int,
        *,
        superseded_by_run: UUID,
        before_sequence: int | None = None,
    ) -> int:
        """Mark the messages of a replaced turn as superseded instead of deleting."""

    @abstractmethod
    async def supersede_run_messages(
        self,
        conversation_id: UUID,
        run_id: UUID,
        *,
        superseded_by_run: UUID,
    ) -> int:
        """Hide the messages a failed attempt appended, keeping the previous turn."""

    @abstractmethod
    async def delete_run(self, run_id: UUID) -> None: ...


class PaperSearchGateway(ABC):
    @abstractmethod
    async def search(self, search_input: PaperSearchInput) -> PaperSearchResult: ...


class PaperDocumentGateway(ABC):
    @abstractmethod
    async def fetch(self, url: str) -> PdfDocument: ...


class ScientificPaperParser(ABC):
    @abstractmethod
    async def parse(
        self,
        path: str,
        *,
        paper_id: str,
        title: str | None = None,
        asset_dir: Path | None = None,
    ) -> ParsedPaperDocument: ...


class TextEmbeddingGateway(ABC):
    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...


class TextRerankerGateway(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]: ...


class TreeVectorStore(ABC):
    @property
    @abstractmethod
    def collection_name(self) -> str: ...

    @abstractmethod
    async def replace_paper(
        self,
        document: ParsedPaperDocument,
        nodes: Sequence[TreeIndexNode],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    @abstractmethod
    async def delete_paper(self, paper_id: str) -> None: ...

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str] | None,
        top_k: int,
        chunks_only: bool,
        node_types: Sequence[TreeNodeType] | None = None,
        parent_ids: Sequence[str] | None = None,
    ) -> list[TreeVectorMatch]: ...

    @abstractmethod
    async def keyword_search(
        self,
        terms: Sequence[str],
        *,
        paper_ids: Sequence[str] | None,
        top_k: int,
        chunks_only: bool = True,
        node_types: Sequence[TreeNodeType] | None = None,
        parent_ids: Sequence[str] | None = None,
    ) -> KeywordSearchResult:
        """Lexical recall over stored text, independent of any vector candidate."""

    @abstractmethod
    async def load_paper_nodes(
        self,
        paper_ids: Sequence[str],
    ) -> list[IndexedTreeNode]: ...

    @abstractmethod
    async def load_nodes(
        self,
        *,
        paper_ids: Sequence[str] | None = None,
        node_ids: Sequence[str] | None = None,
        parent_ids: Sequence[str] | None = None,
        node_types: Sequence[TreeNodeType] | None = None,
    ) -> list[IndexedTreeNode]:
        """Load a bounded node set on demand instead of a whole paper tree."""


class EventPublisher(ABC):
    @abstractmethod
    async def publish(
        self,
        event_type: str,
        payload: dict[str, JsonValue],
    ) -> None: ...
