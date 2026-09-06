import asyncio
import importlib
import threading
from collections.abc import Sequence
from typing import Any, cast

from app.domain.ports import TextRerankerGateway


class SentenceTransformerCrossEncoderReranker(TextRerankerGateway):
    """Lazy local Cross-Encoder adapter backed by sentence-transformers."""

    def __init__(
        self,
        *,
        model_name: str,
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Cross-Encoder model_name cannot be empty")
        if max_length < 32:
            raise ValueError("Cross-Encoder max_length must be at least 32")
        self._model_name = model_name.strip()
        self._max_length = max_length
        self._device = device.strip() if device and device.strip() else None
        self._model: Any | None = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._model_name

    async def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, list(documents))

    def _rerank_sync(self, query: str, documents: list[str]) -> list[float]:
        model = self._load_model()
        try:
            raw_scores = model.predict(
                [(query, document) for document in documents],
                convert_to_numpy=False,
                show_progress_bar=False,
            )
        except Exception as error:
            raise RuntimeError(f"Cross-Encoder inference failed: {error}") from error
        scores = self._to_float_list(raw_scores)
        if len(scores) != len(documents):
            raise RuntimeError(
                "Cross-Encoder returned a different number of scores than documents"
            )
        return scores

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise RuntimeError(self._load_error)
            try:
                module = importlib.import_module("sentence_transformers")
            except ModuleNotFoundError as error:
                self._load_error = (
                    "Cross-Encoder reranking requires the optional 'reranker' dependencies"
                )
                raise RuntimeError(self._load_error) from error
            cross_encoder = cast(Any, module.CrossEncoder)
            options: dict[str, object] = {"max_length": self._max_length}
            if self._device is not None:
                options["device"] = self._device
            try:
                self._model = cross_encoder(self._model_name, **options)
            except Exception as error:
                self._load_error = f"Cross-Encoder model loading failed: {error}"
                raise RuntimeError(self._load_error) from error
            return self._model

    @staticmethod
    def _to_float_list(values: object) -> list[float]:
        if hasattr(values, "tolist"):
            values = cast(Any, values).tolist()
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            values = [values]
        return [float(cast(Any, value)) for value in values]
