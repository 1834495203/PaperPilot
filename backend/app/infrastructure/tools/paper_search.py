from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.papers import (
    PaperSearchAttempt,
    PaperSearchInput,
    PaperSearchResult,
    PaperSource,
)
from app.domain.ports import PaperSearchGateway


@dataclass(frozen=True, slots=True)
class SearchProvider:
    name: PaperSource
    gateway: PaperSearchGateway


class PaperSearchFallbackError(RuntimeError):
    def __init__(self, attempts: list[PaperSearchAttempt]) -> None:
        message = "; ".join(
            f"{item.provider.value}: {item.error_message or item.status}" for item in attempts
        )
        super().__init__(f"All academic search providers failed: {message}")
        self.category = "provider_error"
        self.retryable = any(item.status == "failed" for item in attempts)
        self.status_code = None
        self.retry_after_seconds = None
        self.attempts = attempts


class FallbackPaperSearchGateway(PaperSearchGateway):
    """Use a deterministic provider chain; the model never chooses a source."""

    def __init__(self, providers: Sequence[SearchProvider]) -> None:
        if not providers:
            raise ValueError("At least one paper search provider is required")
        self._providers = tuple(providers)

    async def search(self, search_input: PaperSearchInput) -> PaperSearchResult:
        attempts: list[PaperSearchAttempt] = []
        successful_request = False
        for provider in self._providers:
            try:
                result = await provider.gateway.search(search_input)
            except (ValueError, RuntimeError) as error:
                attempts.append(
                    PaperSearchAttempt(
                        provider=provider.name,
                        status="failed",
                        error_category=str(getattr(error, "category", "tool_error")),
                        error_message=str(error) or type(error).__name__,
                    )
                )
                continue
            successful_request = True
            attempts.extend(result.attempts)
            if result.papers:
                return result.model_copy(update={"attempts": attempts})
        if not successful_request:
            raise PaperSearchFallbackError(attempts)
        return PaperSearchResult(papers=[], provider=None, attempts=attempts)
