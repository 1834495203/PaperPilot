from types import SimpleNamespace

import pytest

from app.infrastructure.rag.reranker import SentenceTransformerCrossEncoderReranker


@pytest.mark.asyncio
async def test_cross_encoder_adapter_scores_query_document_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class _Model:
        def __init__(self, model_name: str, **options: object) -> None:
            calls["model_name"] = model_name
            calls["options"] = options

        def predict(
            self,
            pairs: list[tuple[str, str]],
            **options: object,
        ) -> list[float]:
            calls["pairs"] = pairs
            calls["predict_options"] = options
            return [0.2, 0.9]

    monkeypatch.setattr(
        "app.infrastructure.rag.reranker.importlib.import_module",
        lambda name: SimpleNamespace(CrossEncoder=_Model),
    )
    reranker = SentenceTransformerCrossEncoderReranker(
        model_name="test/model",
        max_length=384,
        device="cpu",
    )

    scores = await reranker.rerank("query", ["first", "second"])

    assert scores == [0.2, 0.9]
    assert calls["model_name"] == "test/model"
    assert calls["options"] == {"max_length": 384, "device": "cpu"}
    assert calls["pairs"] == [("query", "first"), ("query", "second")]
