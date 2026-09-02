import math
from collections.abc import Sequence

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.domain.ports import TextEmbeddingGateway


class OpenAITextEmbeddingGateway(TextEmbeddingGateway):
    """Generate normalized vectors through an OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        *,
        model: str,
        api_key: SecretStr,
        base_url: str | None,
        dimensions: int | None = None,
    ) -> None:
        if not api_key.get_secret_value():
            raise ValueError("EMBEDDING_API_KEY is required for paper ingestion")
        self._client = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            dimensions=dimensions,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await self._client.aembed_documents(list(texts))
        return [self._normalize(vector) for vector in vectors]

    async def embed_query(self, text: str) -> list[float]:
        vector = await self._client.aembed_query(text)
        return self._normalize(vector)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            raise ValueError("Embedding provider returned a zero vector")
        return [value / magnitude for value in vector]
